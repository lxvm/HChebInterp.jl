module AdaptChebInterp

using LinearAlgebra

using FastChebInterp: chebpoint, chebpoints, chebinterp, ChebPoly

export adaptchebinterp, fastadaptchebinterp

# desired features
# - Reusing function evaluations at hypercube corners
# - Parallelization/vectorization of function evaluations at each level of refinement
# - various error estimation options allowing quad/oct-tree refinement,
#   refinement one dimension at a time, a la HCubature, and everything in between

struct Panel3{N,Td,S}
    lb::Td # left endpoint
    ub::Td # right endpoint
    val::Int # if nonzero, index of val in valtree
    next::S # indices of subpanels
end

Panel = Panel2

struct PanelPoly1{N,T,Td} <: Function
    valtree::Vector{ChebPoly{N,T,Td}}
    searchtree::Vector{Panel{Td}}
    lb::Td
    ub::Td
    initdivs::NTuple{N,Int}
end

PanelPoly = PanelPoly1

function (p::PanelPoly{N,T,Td})(x_) where {N,T,Td}
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


function chebpoints!(p::Array{T,N}, order::NTuple{N,Int}, lb, ub) where {T,N}
    size(p) == order .+ 1 || error("wrong chebpoint array size")
    @inbounds for i in Iterators.product(map(n -> n==0 ? (1:1) : (0:n), order)...)
        p[(i .+ 1)...] = chebpoint(CartesianIndex(i), order, lb, ub)
    end
    p
end


function fastadaptchebinterp(f, a::A, b::B; order=15, atol=0, rtol=0, norm=norm, maxevals=typemax(Int), initdiv=1, abs=abs, ncoeffs=3) where {A,B}
    @assert ncoeffs <= order+1 "insufficient coefficients to estimate error"
    T = float(promote_type(A, B))
    fastadaptchebinterp_(f, T(a), T(b), order, atol, rtol, norm, maxevals, initdiv, abs, ncoeffs)
end

function fastadaptchebinterp_(f, a::T, b::T, order, atol, rtol_, norm, maxevals, initdiv, abs, ncoeffs) where T
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
            E = fastevalerror(c, abs, order, ncoeffs)

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


function fastevalerror(c::ChebPoly{N}, abs, order, ncoeffs) where N
    # idea: compare size (better: rate of decay) of the Chebyshev coefficients
    n = max(0, ncoeffs + length(c.coefs) - order - 1) # FastChebInterp truncates coefficients under a tolerance
    E = (n-ncoeffs)*eps(abs(c.coefs[end])) # initialize error, possibly offset by truncation
    for i in (order+2-ncoeffs):(order+1-ncoeffs+n)
        E += abs(c.coefs[i])
    end
    return E
end


end # module AdaptChebInterp
