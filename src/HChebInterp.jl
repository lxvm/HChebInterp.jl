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

export hchebinterp, SpectralError, HAdaptError, BatchFunction

struct TreePoly{N,V,T,Td,Tx} <: Function
    valtree::Vector{Array{V,N}}
    funtree::Vector{ChebPoly{N,T,Td}}
    searchtree::Vector{Vector{Int}}
    lb::SVector{N,Tx}
    ub::SVector{N,Tx}
    initdiv::Int
end

_indomain(lb::T, ub::T, x::T) where {T<:SVector{1}} = lb[1] <= x[1] <= ub[1]
function _indomain(lb::T, ub::T, x::T) where {N,T<:SVector{N}}
    return (lb[N] <= x[N] <= ub[N]) && _indomain(pop(lb), pop(ub), pop(x))
end
_indomain(args...) = _indomain(promote(args...)...)
function indomain(interp::ChebPoly{N}, x::SVector{N}) where {N}
    return _indomain(interp.lb, interp.ub, x)
end

# x = a + (1+t)*(b-a)/2, t∈[-1,1]
# t = (x-a)*2/(b-a) - 1, x∈[a,b]
function (p::TreePoly{N,T,V,Td,Tx})(x::SVector{N,Tx}) where {N,T,V,Td,Tx}
    t = map((a, b, x) -> (x-a)*2/(b-a)-1, p.lb, p.ub, x)
    for i in 1:p.initdiv^N
        fun = p.funtree[i]
        children = p.searchtree[i]
        indomain(fun, t) || continue
        while true
            isempty(children) && return convert(V, fun(t))
            for c in children
                indomain(p.funtree[c], t) || continue
                fun = p.funtree[c]
                children = p.searchtree[c]
                break
            end
        end
    end
    throw(ArgumentError("$x not in domain $(p.lb) to $(p.ub)"))
end
(p::TreePoly{N,T,V,Td,Tx})(x) where {N,T,V,Td,Tx} = p(SVector{N,Tx}(x))

# chebpoint does not use precision of endpoints in current release v1.2
function _chebpoint(i::CartesianIndex{N}, order::NTuple{N,Int}, lb::SVector{N}, ub::SVector{N}) where {N}
    T = typeof(float(one(eltype(lb)) * one(eltype(ub))))
    @. lb + (1 + cos(T($SVector($Tuple(i))) * π / $SVector(ifelse.(iszero.(order),2,order)))) * (ub - lb) * $(T(0.5))
end
_chebpoint(i,order,lb,ub) = chebpoint(i,order,lb,ub)
_chebpoint(i,order,lb::Number,ub::Number) = only(_chebpoint(i,order,SVector(lb),SVector(ub)))
function chebpoints!(p::Array{T,N}, order::NTuple{N,Int}, a::T, b::T) where {T,N}
    for i in CartesianIndices(map(n -> n==0 ? (1:1) : (0:n), order))
        p[CartesianIndex(i.I .+ 1)] = _chebpoint(i,order,a,b)
    end
    p
end

function _chebinterp(data::AbstractArray{<:Number}, args...; kws...)
    u = oneunit(eltype(data))
    c = chebinterp(data/u, args...; kws...)
    return ChebPoly(u*c.coefs, c.lb, c.ub)
end
function _chebinterp(data::AbstractArray{<:SVector}, args...; kws...)
    u = oneunit(eltype(eltype(data)))
    chebinterp(data/u, args...; kws...)
    return ChebPoly(u*c.coefs, c.lb, c.ub)
end
function _chebinterp(data::AbstractArray{<:SArray}, args...; kws...)
    c = chebinterp(reinterpret(SVector{length(eltype(data)), typeof(one(eltype(eltype(data))))}, data), args...; kws...)
    return ChebPoly(Array(reinterpret(eltype(data), c.coefs)), c.lb, c.ub)
end
function _chebinterp(data::AbstractArray{<:AbstractArray{T,N}}, args...; kws...) where {T,N}
    u = oneunit(T)
    next = iterate(data)
    isnothing(next) && throw(ArgumentError("no elements in array"))
    item, state = next
    s = size(item)
    next = iterate(data, state)
    while !isnothing(next)
        item, state = next
        s == size(item) || throw(ArgumentError("array elements must all be of same size"))
        next = iterate(data, state)
    end
    len = prod(s)
    c = chebinterp(SVector{len}.(./(data, u)), args...; kws...)
    return ChebPoly(u*Array(reinterpret(SArray{Tuple{s...},T,N,len}, c.coefs)), c.lb, c.ub)
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

"""
    BatchFunction(f, [x])

Wrapper for an out-of-place function of the form `f.(x)`.
Optionally provide a buffer `x` to store the input points of the right shape, namely
`all(size(x) .+ 1 .== order)`
"""
struct BatchFunction{F,X}
    f::F
    x::X
end
BatchFunction(f) = BatchFunction(f, nothing)

_oftype(y, x) = oftype(y, x)
_oftype(y::T, x::MVector{1,T}) where {T} = oftype(y, only(x))

function evalnext!(nextval, nextfun, criterion::HAdaptError, f::BatchFunction, lb, ub, val, fun::ChebPoly{n,T,Td}, order, atol, rtol, norm, droptol) where {n,T,Td}
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
        x = _oftype(lb, ma)
        y = _oftype(ub, mb)
        # this is shorter and has unrolled loops, but somehow creates a type instability:
        # x = SVector(ntuple(i -> a[i]+(c[i]-1)*Δ[i], Val{n}()))
        # y = SVector(ntuple(i -> c[i]==initdiv ? b[i] : a[i]+c[i]*Δ[i], Val{n}()))
        push!(nextval, f.f(chebpoints!(f.x, order, x, y)))
        push!(nextfun, _chebinterp(nextval[end], x, y; tol=droptol))

        # evaluate the error on a dense grid
        p = Iterators.product(ntuple(m -> range(x[m], y[m], length=order[m]*criterion.n), Val{n}())...)
        E = maximum(r_ -> (r = SVector{n,Td}(r_); norm(fun(r) - nextfun[end](r))), p)
        converged &= E < max(tol, rtol*maximum(norm, nextval[end]))
    end
    converged
end

function evalnext!(nextval, nextfun, criterion::SpectralError, f::BatchFunction, lb,ub, val, fun::ChebPoly{n,T,Td}, order, atol, rtol, norm, droptol) where {n,T,Td}
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
        x = _oftype(lb, ma)
        y = _oftype(ub, mb)
        # this is shorter and has unrolled loops, but somehow creates a type instability:
        # x = SVector(ntuple(i -> a[i]+(c[i]-1)*Δ[i], Val{n}()))
        # y = SVector(ntuple(i -> c[i]==initdiv ? b[i] : a[i]+c[i]*Δ[i], Val{n}()))
        push!(nextval, f.f(chebpoints!(f.x, order, x, y)))
        push!(nextfun, _chebinterp(nextval[end], x, y; tol=droptol))
    end
    all(dimsconverged)
end


function hchebinterp_(criterion::AbstractAdaptCriterion, f::BatchFunction, a, b, order, atol_, rtol_, norm, maxevals, initdiv, droptol)
    maxevals < 0 && throw(ArgumentError("invalid negative maxevals"))
    initdiv < 1 && throw(ArgumentError("initdiv must be positive"))


    Δ = (b-a) / initdiv
    b1 = initdiv == 1 ? b : a+Δ
    valtree = [f.f(chebpoints!(f.x, order, a, b))]
    funtree = [_chebinterp(valtree[1], a, b1; tol=droptol)]
    searchtree = [Int[]]

    atol = something(atol_, zero(norm(first(valtree[1]))))
    rtol = something(rtol_, iszero(atol) ? sqrt(eps(one(atol))) : zero(one(atol)))

    ma = MVector(a)
    mb = MVector(b)
    n = length(a)

    if initdiv > 1 # initial box divided by initdiv along each dimension
        skip = true # skip the first box, which we already added
        @inbounds for c in CartesianIndices(ntuple(i->Base.OneTo(initdiv), Val{n}())) # Val ntuple loops are unrolled
            if skip; skip=false; continue; end
            for i = 1:n
                ma[i] = a[i]+(c[i]-1)*Δ[i]
                mb[i] = c[i]==initdiv ? b[i] : a[i]+c[i]*Δ[i]
            end
            x = _oftype(a, ma)
            y = _oftype(b, mb)
            # this is shorter and has unrolled loops, but somehow creates a type instability:
            # x = SVector(ntuple(i -> a[i]+(c[i]-1)*Δ[i], Val{n}()))
            # y = SVector(ntuple(i -> c[i]==initdiv ? b[i] : a[i]+c[i]*Δ[i], Val{n}()))
            push!(valtree, f.f(chebpoints!(f.x, order, x, y)))
            push!(funtree, _chebinterp(valtree[end], x, y; tol=droptol))
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
            converged = evalnext!(nextval, nextfun, criterion, f, a,b, valtree[i], funtree[i], order, atol, rtol, norm, droptol)
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
    TreePoly(valtree, funtree, searchtree, a isa Number ? SVector(a) : a, b isa Number ? SVector(b) : b, initdiv)
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
function hchebinterp(f::BatchFunction, a::SVector{n,T}, b::SVector{n,T}; criterion=SpectralError(), order=15, atol=nothing, rtol=nothing, norm=norm, maxevals=typemax(Int), initdiv=1, droptol=0) where {n,T}
    @assert one(T) isa Real "a and b must be real vectors"
    s = Diagonal((b-a)/2)
    e = ones(SVector{n,typeof(one(T))})
    t2x = t -> a + s * (e + t)
    ord = fill_ntuple(order, n)
    t = Array{typeof(e),n}(undef, ord .+ 1)
    x = isnothing(f.x) ? similar(t, typeof(a)) : f.x
    @assert size(x) == size(t)
    g = BatchFunction(t -> f.f(map!(t2x, x, t)), t)
    p = hchebinterp_(criterion, g, -e, e, ord, atol, rtol, norm, maxevals, initdiv, droptol)
    return TreePoly(p.valtree, p.funtree, p.searchtree, a, b, p.initdiv)
end

function hchebinterp(f, a, b; kws...)
    (n = length(a)) == length(b) || throw(ArgumentError("a and b must be the same length"))
    T = float(promote_type(eltype(a),eltype(b)))
    g = if a isa Number
        if f isa BatchFunction
            BatchFunction(x -> f.f(reinterpret(T, x)), isnothing(f.x) ? f.x : reinterpret(SVector{n,T}, f.x))
        else
            BatchFunction(x -> f.(reinterpret(T, x)))
        end
    else
        f isa BatchFunction ? f : BatchFunction(x -> f.(x))
    end
    return hchebinterp(g, SVector{n,T}(a), SVector{n,T}(b); kws...)
end

end
