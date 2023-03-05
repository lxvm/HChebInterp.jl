# desired features
# - Reusing function evaluations at hypercube corners
# - Parallelization/vectorization of function evaluations at each level of refinement
# - various error estimation options allowing quad/oct-tree refinement,
#   refinement one dimension at a time, a la HCubature, and everything in between

"""
A package for h-adaptive Chebyshev interpolation of 1D functions using
[`FastChebInterp.jl`](https://github.com/stevengj/FastChebInterp.jl).
Algorithms based on work by [Kaye et al.](http://arxiv.org/abs/2211.12959).
See the tests for examples.
"""
module HChebInterp

using LinearAlgebra

using StaticArrays
using FastChebInterp: chebpoint, chebpoints, chebinterp, ChebPoly

export hchebinterp, SpectralError, HAdaptError

struct Panel{Td}
    a::Td # left endpoint
    b::Td # right endpoint
    val::Int # if nonzero, index of val in valtree
    lt::Int # if nonzero, index of left subpanel
    gt::Int # if nonzero, index of right subpanel
end


struct PanelPoly{T,Td} <: Function
    valtree::Vector{ChebPoly{1,T,Td}}
    searchtree::Vector{Panel{Td}}
    lb::Td
    ub::Td
    initdiv::Int
end

function (p::PanelPoly{T,Td})(x_) where {T,Td}
    x = convert(Td, x_)
    p.lb <= x <= p.ub || throw(ArgumentError("x is outside of the domain"))
    for i in 1:p.initdiv
        panel = p.searchtree[i]
        if panel.a <= x <= panel.b
            while iszero(panel.val)
                panel = 2x <= panel.a+panel.b ? p.searchtree[panel.lt] : p.searchtree[panel.gt]
            end
            return p.valtree[panel.val](x)
        end
    end
    error("panel in domain not found")
end


function chebpoints!(p::Vector{Float64}, order::Int, a::Ta, b::Tb) where {Ta,Tb}
    order_ = (order,)
    a_ = SVector{1,Ta}(a)
    b_ = SVector{1,Tb}(b)
    @inbounds for i in 0:order
        p[i+1] = chebpoint(CartesianIndex(i), order_, a_, b_)[1]
    end
    p
end

"""
    AbstractAdaptCriterion

Abstract supertype for error criteria for adaptive refinement.
"""
abstract type AbstractAdaptCriterion end

"""
    SpectralError(; n=3, abs=abs)

Estimate the error of the interpolant by as the sum of the norm of the last `n`
Chebyshev coefficients. Use `abs` to compute the norm of each coefficient.
"""
struct SpectralError{A} <: AbstractAdaptCriterion
    n::Int
    abs::A
end

SpectralError(; n=10, abs=abs) = SpectralError(n, abs)

"""
    HAdaptError(; n=10)

Estimate the error of the interpolant by dividing the panel into two, computing
interpolants on the subpanels, and computing the maximum error between
interpolants at `n*p` equispaced points, where `p` is the number of points used
to compute each interpolant.
"""
struct HAdaptError <: AbstractAdaptCriterion
    n::Int
end
HAdaptError(; n=10) = HAdaptError(n)



function hchebinterp_(criterion::HAdaptError, f, a::T, b::T, order, atol, rtol_, norm, maxevals, initdiv) where T
    npoints = criterion.n
    rtol = rtol_ == 0 == atol ? sqrt(eps(T)) : rtol_
    (rtol < 0 || atol < 0) && throw(ArgumentError("invalid negative tolerance"))
    maxevals < 0 && throw(ArgumentError("invalid negative maxevals"))
    initdiv < 1 && throw(ArgumentError("initdiv must be positive"))


    # first panel
    r = range(a, b; length=initdiv+1)
    lb, ub = r[1], r[2]
    p = chebpoints(order, lb, ub)
    fp = f.(p)
    c = chebinterp(fp, lb, ub)
    valtree = [c]
    searchtree = [Panel(r[1],r[2],1,0,0)]
    numevals = evals_per_panel = order + 1

    # remaining panels
    for i in 2:initdiv
        lb, ub = r[i], r[i+1]
        chebpoints!(p, order, lb, ub)
        fp .= f.(p)
        c = chebinterp(fp, lb, ub)
        push!(valtree, c)
        push!(searchtree, Panel(r[i],r[i+1],i,0,0))
        numevals += evals_per_panel
    end

    nvals = npanels = initdiv
    val_idx = collect(1:initdiv)
    val_idx_ = Int[]
    pan_idx = collect(1:initdiv)
    pan_idx_ = Int[]

    while true
        npanels_ = npanels
        panels = view(searchtree, pan_idx)
        for (i, (idx, panel)) in enumerate(zip(val_idx, panels))
            numevals > maxevals && break

            c = valtree[idx]
            lb = only(c.lb)
            ub = only(c.ub)
            mid = (lb+ub)/2

            chebpoints!(p, order, lb, mid)
            fp .= f.(p)
            lc = chebinterp(fp, lb, mid)
            lf = maximum(norm, fp)
            
            chebpoints!(p, order, mid, ub)
            fp .= f.(p)
            rc = chebinterp(fp, mid, ub)
            rf = maximum(norm, fp)
            
            numevals += 2evals_per_panel

            valtree[idx] = lc
            push!(valtree, rc)
            nvals += 1
            push!(searchtree, Panel(lb, mid, idx, 0, 0))
            push!(searchtree, Panel(mid, ub, nvals, 0, 0))
            npanels += 2
            panels[i] = Panel(panel.a, panel.b, 0, npanels-1, npanels)
            
            E = evalhadapterror(c, lc, rc, lb, mid, ub, order, norm, npoints)
            if E > max(atol, rtol*max(lf, rf))
                # these lines indicate where the algorithm should refine again
                push!(val_idx_, idx, nvals)
                push!(pan_idx_, npanels-1, npanels)
            else
                npanels_ += 2
            end
        end
        npanels_ == npanels && break
        resize!(val_idx, length(val_idx_))
        val_idx .= val_idx_
        resize!(val_idx_, 0)
        resize!(pan_idx, length(pan_idx_))
        pan_idx .= pan_idx_
        resize!(pan_idx_, 0)
    end
    PanelPoly(valtree, searchtree, a, b, initdiv)
end

function evalhadapterror(c, lc, rc, a, mid, b, order, norm, npoints)
    # idea: compare the maximum difference on a dense grid, which fails
    # when the polynomial degree of the true function is too high
    E = norm(c(a) - lc(a))
    for x in range(a, mid; length=npoints*(order+1))
        x == a && continue
        E = max(norm(c(x) - lc(x)), E)
    end
    for x in range(mid, b; length=npoints*(order+1))
        x == mid && continue
        E = max(norm(c(x) - rc(x)), E)
    end
    E
end


function hchebinterp_(criterion::SpectralError, f, a::T, b::T, order, atol, rtol_, norm, maxevals, initdiv) where T
    ncoeffs = criterion.n
    abs = criterion.abs
    rtol = rtol_ == 0 == atol ? sqrt(eps(T)) : rtol_
    (rtol < 0 || atol < 0) && throw(ArgumentError("invalid negative tolerance"))
    maxevals < 0 && throw(ArgumentError("invalid negative maxevals"))
    initdiv < 1 && throw(ArgumentError("initdiv must be positive"))


    # first panel
    r = range(a, b; length=initdiv+1)
    lb, ub = r[1], r[2]
    p = chebpoints(order, lb, ub)
    fp = f.(p)
    c = chebinterp(fp, lb, ub)
    nrmtree = [maximum(norm, fp)]
    valtree = [c]
    searchtree = [Panel(r[1],r[2],1,0,0)]
    numevals = evals_per_panel = order + 1

    # remaining panels
    for i in 2:initdiv
        lb, ub = r[i], r[i+1]
        chebpoints!(p, order, lb, ub)
        fp .= f.(p)
        c = chebinterp(fp, lb, ub)
        push!(valtree, c)
        push!(nrmtree, maximum(norm, fp))
        push!(searchtree, Panel(r[i],r[i+1],i,0,0))
        numevals += evals_per_panel
    end

    nvals = npanels = initdiv
    val_idx = collect(1:initdiv)
    val_idx_ = Int[]
    panels = view(searchtree, 1:initdiv)

    while true
        npanels_ = npanels
        for (i, (idx, panel)) in enumerate(zip(val_idx, panels))
            numevals > maxevals && break
            
            c = valtree[idx]
            E = evalspectralerror(c, abs, order, ncoeffs)

            if E > max(atol, rtol*nrmtree[idx])
                
                lb = only(c.lb)
                ub = only(c.ub)
                mid = (lb+ub)/2

                chebpoints!(p, order, lb, mid)
                fp .= f.(p)
                valtree[idx] = chebinterp(fp, lb, mid)
                nrmtree[idx] = maximum(norm, fp)

                chebpoints!(p, order, mid, ub)
                fp .= f.(p)
                push!(valtree, chebinterp(fp, mid, ub))
                push!(nrmtree, maximum(norm, fp))
                
                numevals += 2evals_per_panel

                nvals += 1
                push!(val_idx_, idx, nvals)
                push!(searchtree, Panel(lb, mid, idx, 0, 0))
                push!(searchtree, Panel(mid, ub, nvals, 0, 0))
                panels[i] = Panel(panel.a, panel.b, 0, npanels+1, npanels+2)
                npanels += 2
            end
        end
        npanels_ == npanels && break
        resize!(val_idx, length(val_idx_))
        val_idx .= val_idx_
        resize!(val_idx_, 0)
        panels = view(searchtree, (npanels_+1):npanels)
    end
    PanelPoly(valtree, searchtree, a, b, initdiv)
end


function evalspectralerror(c::ChebPoly{1}, abs, order, ncoeffs)
    # idea: compare size (better: rate of decay) of the Chebyshev coefficients
    n = max(0, ncoeffs + length(c.coefs) - order - 1) # FastChebInterp truncates coefficients under a tolerance
    E = (n-ncoeffs)*eps(abs(c.coefs[end])) # initialize error, possibly offset by truncation
    for i in (order+2-ncoeffs):(order+1-ncoeffs+n)
        E += abs(c.coefs[i])
    end
    return E
end

"""
    hchebinterp(f, a, b, [criterion=SpectralError()]; order=15, atol=0, rtol=0, norm=norm, maxevals=typemax(Int), initdiv=1)
    
Return a piecewise polynomial interpolant of `f` on the interval ``[a,b]`` of
degree `order` that is pointwise accurate to the requested tolerances. Uses
`criterion::AbstractAdaptCriterion` to estimate the interpolant error for
h-adaptation. If `HAdaptError()` is used as the criterion, it may be appropriate
to reduce the `order` to 4 to avoid unnecessary function evaluations.
"""
function hchebinterp(f, a::A, b::B, criterion=SpectralError(); order=15, atol=0, rtol=0, norm=norm, maxevals=typemax(Int), initdiv=1) where {A,B}
    T = float(promote_type(A, B))
    hchebinterp_(criterion, f, T(a), T(b), order, atol, rtol, norm, maxevals, initdiv)
end

end