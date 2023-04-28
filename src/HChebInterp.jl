# desired features
# - Reusing function evaluations at hypercube corners
# - Parallelization/vectorization of function evaluations at each level of refinement
# - various error estimation options allowing quad/oct-tree refinement,
#   refinement one dimension at a time, a la HCubature, and everything in between

"""
A package for h-adaptive Chebyshev interpolation of N-D functions using
[`FastChebInterp.jl`](https://github.com/stevengj/FastChebInterp.jl).
Algorithms based on work by [Kaye et al.](http://arxiv.org/abs/2211.12959).
See the tests for examples.
"""
module HChebInterp

using LinearAlgebra

using StaticArrays
using FastChebInterp: chebpoint, chebpoints, chebinterp, ChebPoly

export hchebinterp, SpectralError, HAdaptError

struct TreePoly{N,T,Td} <: Function
    valtree::Vector{Array{T,N}}
    funtree::Vector{ChebPoly{N,T,Td}}
    searchtree::Vector{Vector{Int}}
    lb::SVector{N,Td}
    ub::SVector{N,Td}
    initdiv::Int
end

@generated function indomain(interp::ChebPoly{N}, x::SVector{N,<:Real}) where N
    :(@inbounds Base.Cartesian.@nall $N d -> (interp.lb[d] <= x[d] <= interp.ub[d]))
    # x0 = @. (x - interp.lb) * 2 / (interp.ub - interp.lb) - 1
    # all(abs.(x0) .≤ 1)
end

function (p::TreePoly{N,T,Td})(x_) where {N,T,Td}
    x = SVector{N,Td}(x_)
    for i in 1:p.initdiv^N
        fun = p.funtree[i]
        children = p.searchtree[i]
        indomain(fun, x) || continue
        while true
            isempty(children) && return fun(x)
            for c in children
                indomain(p.funtree[c], x) || continue
                fun = p.funtree[c]
                children = p.searchtree[c]
                break
            end
        end
    end
    throw(ArgumentError("$x not in domain $(p.lb) to $(p.ub)"))
end


function chebpoints!(p::Array{<:SVector{N},N}, order::NTuple{N,Int}, a::SVector{N}, b::SVector{N}) where N
    for i in CartesianIndices(map(n -> n==0 ? (1:1) : (0:n), order))
        p[CartesianIndex(i.I .+ 1)] = chebpoint(i,order,a,b) 
    end
    p
end

"""
    AbstractAdaptCriterion

Abstract supertype for error criteria for adaptive refinement.
"""
abstract type AbstractAdaptCriterion end

"""
    SpectralError(; n=3)

Estimate the error of the interpolant by as the sum of the norm of the last `n`
Chebyshev coefficients. Use `abs` to compute the norm of each coefficient.
"""
struct SpectralError <: AbstractAdaptCriterion
    n::Int
end

SpectralError(; n=3) = SpectralError(n)

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

# layer to intercept function evaluation for parallelization
batcheval(f, x, ::Type) = f.(x)
batcheval(f, x, ::Type{<:Real}) = f.(only.(x))
# TODO parallelize all function evaluations in each queue

function evalnext!(nextval, nextfun, criterion::HAdaptError, f, val, fun::ChebPoly{n,T,Td}, order, atol, rtol, norm, droptol, Ta) where {n,T,Td}
    tol = max(atol, rtol*maximum(norm, val))

    a = fun.lb
    b = fun.ub
    Δ = (b-a) / 2

    ma = MVector(a)
    mb = MVector(b)

    converged = true
    @inbounds for c in CartesianIndices(ntuple(i->Base.OneTo(2), Val{n}())) # Val ntuple loops are unrolled
        for i = 1:n
            ma[i] = a[i]+(c[i]-1)*Δ[i]
            mb[i] = c[i]==2 ? b[i] : a[i]+c[i]*Δ[i]
        end
        x = SVector(ma)
        y = SVector(mb)
        # this is shorter and has unrolled loops, but somehow creates a type instability:
        # x = SVector(ntuple(i -> a[i]+(c[i]-1)*Δ[i], Val{n}()))
        # y = SVector(ntuple(i -> c[i]==initdiv ? b[i] : a[i]+c[i]*Δ[i], Val{n}()))
        push!(nextval, batcheval(f, chebpoints(order, x, y), Ta))
        push!(nextfun, chebinterp(nextval[end], x, y; tol=droptol))

        # evaluate the error on a dense grid
        p = Iterators.product(ntuple(m -> range(x[m], y[m], length=order[m]*criterion.n), Val{n}())...)
        E = maximum(r_ -> (r = SVector{n,Td}(r_); norm(fun(r) - nextfun[end](r))), p)
        converged &= E < max(tol, rtol*maximum(norm, nextval[end]))
    end
    converged
end

function evalnext!(nextval, nextfun, criterion::SpectralError, f, val, fun::ChebPoly{n,T,Td}, order, atol, rtol, norm, droptol, Ta) where {n,T,Td}
    tol = max(atol, rtol*maximum(norm, val))

    a = fun.lb
    b = fun.ub
    Δ = (b-a) / 2

    ma = MVector(a)
    mb = MVector(b)

    dimsconverged = ntuple(Val{n}()) do i
        # assuming size(fun.coefs) = order
        idx = CartesianIndices(ntuple(j -> j==i ? (2+order[i]-criterion.n:size(fun.coefs,i)) : axes(fun.coefs, j), Val{n}()))
        tol > maximum(sum(norm, @view fun.coefs[idx]; dims=i))
    end

    newsize = map(v -> v ? 1 : 2, dimsconverged)
    @inbounds for c in CartesianIndices(ntuple(i->Base.OneTo(newsize[i]), Val{n}())) # Val ntuple loops are unrolled
        for i = 1:n
            ma[i] = a[i]+(c[i]-1)*Δ[i]
            mb[i] = c[i]==newsize[i] ? b[i] : a[i]+c[i]*Δ[i]
        end
        x = SVector(ma)
        y = SVector(mb)
        # this is shorter and has unrolled loops, but somehow creates a type instability:
        # x = SVector(ntuple(i -> a[i]+(c[i]-1)*Δ[i], Val{n}()))
        # y = SVector(ntuple(i -> c[i]==initdiv ? b[i] : a[i]+c[i]*Δ[i], Val{n}()))
        push!(nextval, batcheval(f, chebpoints(order, x, y), Ta))
        push!(nextfun, chebinterp(nextval[end], x, y; tol=droptol))
    end
    all(dimsconverged)
end


function hchebinterp_(criterion::AbstractAdaptCriterion, f, a::T, b::T, order, atol, rtol_, norm, maxevals, initdiv, droptol, Ta) where {n,T<:SVector{n}}
    rtol = rtol_ == 0 == atol ? sqrt(eps(eltype(T))) : rtol_
    (rtol < 0 || atol < 0) && throw(ArgumentError("invalid negative tolerance"))
    maxevals < 0 && throw(ArgumentError("invalid negative maxevals"))
    initdiv < 1 && throw(ArgumentError("initdiv must be positive"))


    Δ = (b-a) / initdiv
    b1 = initdiv == 1 ? b : a+Δ
    valtree = [batcheval(f, chebpoints(order, a, b1), Ta)]
    funtree = [chebinterp(valtree[1], a, b1; tol=droptol)]
    searchtree = [Int[]]

    ma = MVector(a)
    mb = MVector(b)

    if initdiv > 1 # initial box divided by initdiv along each dimension
        skip = true # skip the first box, which we already added
        @inbounds for c in CartesianIndices(ntuple(i->Base.OneTo(initdiv), Val{n}())) # Val ntuple loops are unrolled
            if skip; skip=false; continue; end
            for i = 1:n
                ma[i] = a[i]+(c[i]-1)*Δ[i]
                mb[i] = c[i]==initdiv ? b[i] : a[i]+c[i]*Δ[i]
            end
            x = SVector(ma)
            y = SVector(mb)
            # this is shorter and has unrolled loops, but somehow creates a type instability:
            # x = SVector(ntuple(i -> a[i]+(c[i]-1)*Δ[i], Val{n}()))
            # y = SVector(ntuple(i -> c[i]==initdiv ? b[i] : a[i]+c[i]*Δ[i], Val{n}()))
            push!(valtree, batcheval(f, chebpoints(order, x, y), Ta))
            push!(funtree, chebinterp(valtree[end], x, y; tol=droptol))
            push!(searchtree, Int[])
        end
    end

    l = initdiv^length(a)
    evalsperbox = prod(order)

    queue = Int[]
    nextqueue = collect(1:l)

    nextval = eltype(valtree)[]
    nextfun = eltype(funtree)[]

    while !isempty(nextqueue)
        if l*evalsperbox > maxevals
            @warn "maxevals exceeded"
            break
        end

        copy!(queue, nextqueue)
        empty!(nextqueue)
        for i in queue
            empty!(nextval)
            empty!(nextfun)
            converged = evalnext!(nextval, nextfun, criterion, f, valtree[i], funtree[i], order, atol, rtol, norm, droptol, Ta)
            for (val, fun) in zip(nextval, nextfun)
                l += 1
                push!(valtree, val)
                push!(funtree, fun)
                push!(searchtree, Int[])
                push!(searchtree[i], l)
                !converged && push!(nextqueue, l)
            end
        end
    end
    TreePoly(valtree, funtree, searchtree, a, b, initdiv)
end


fill_ntuple(e::Union{Number,Val}, N) = ntuple(_ -> e, N)
fill_ntuple(e::Tuple, _) = e
fill_ntuple(e::AbstractArray, _) = tuple(e...)

to_svec(a::SVector{n,T}, b::SVector{n,T}) where {n,T<:Real} = (a,b)
to_svec(a::Tuple{Vararg{Real,n}}, b::Tuple{Vararg{Real,n}}) where n =
to_svec(SVector{n}(float.(a)), SVector{n}(float.(b)))
function to_svec(a::AbstractVector{T}, b::AbstractVector{S}) where {T,S}
    length(a) == length(b) || throw(DimensionMismatch("endpoints $a and $b must have the same length"))
    F = float(promote_type(T, S))
    SVector{length(a),F}(a), SVector{length(a),F}(b)
end
function to_svec(a::T, b::S) where {T<:Real,S<:Real}
    F = float(promote_type(T,S))
    (SVector{1,F}(a), SVector{1,F}(b))
end

"""
    hchebinterp(f, a, b, [criterion=SpectralError()]; order=15, atol=0, rtol=0, norm=norm, maxevals=typemax(Int), initdiv=1)
    
Return a piecewise polynomial interpolant of `f` on the interval ``[a,b]`` of
degree `order` that is pointwise accurate to the requested tolerances. Uses
`criterion::AbstractAdaptCriterion` to estimate the interpolant error for
h-adaptation. If `HAdaptError()` is used as the criterion, it may be appropriate
to reduce the `order` to 4 to avoid unnecessary function evaluations.
"""
function hchebinterp(f, a_, b_; criterion=SpectralError(), order=15, atol=0, rtol=0, norm=norm, maxevals=typemax(Int), initdiv=1, droptol=0)
    a, b = to_svec(a_, b_)
    hchebinterp_(criterion, f, a, b, fill_ntuple(order, length(a_)), atol, rtol, norm, maxevals, initdiv, droptol, typeof(a_))
end

end