# TODO
# - batch multiple panels at a time while reusing nodes shared by adjacent panels

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
    ninitdiv::Int
end

function treesearch(callback, valtree, funtree, searchtree, x, ninitdiv)
    for i in 1:min(ninitdiv, length(funtree))
        val = valtree[i]
        fun = funtree[i]
        children = searchtree[i]
        indomain(fun, x) || continue
        r = callback(fun, val, children)
        r === nothing || return r
        while true
            found = false
            for c in children
                indomain(funtree[c], x) || continue
                found = true
                val = valtree[c]
                fun = funtree[c]
                children = searchtree[c]
                break
            end
            if found
                r = callback(fun, val, children)
                r === nothing || return r
                continue
            end
            return nothing
        end
    end
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
    r = treesearch(p.valtree, p.funtree, p.searchtree, t, p.ninitdiv) do fun, val, children
        isempty(children) ? convert(V, fun(t)) : nothing
    end
    r === nothing && throw(ArgumentError("$x not in domain $(p.lb) to $(p.ub)"))
    return r
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
    c = chebinterp(data/u, args...; kws...)
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
    BatchFunction(f, [x::AbstractArray]; max_batch::Integer=typemax(Int))

Wrapper for an out-of-place function of the form `f.(x)`, where the input `x`
will be a mutable vector with a similar element type to the input domain.
Optionally provide a resizeable vector `x` to store the input points.

!!! note "HChebInterp v1.2"
    This version is needed to set the `max_batch` keyword. Previously one panel was
    evaluated at a time.

The keyword `max_batch` sets a soft limit on the number of points to pass to the function.
In practice, the smallest number of complete panels with a number of points exceeding
`max_batch` is used.
"""
struct BatchFunction{F,X<:AbstractVector}
    f::F
    x::X
    max_batch::Int
end
BatchFunction(f; max_batch::Integer=typemax(Int)) = BatchFunction(f, Nothing[], max_batch)
BatchFunction(f, x::AbstractArray; max_batch::Integer=typemax(Int)) =
    BatchFunction(f, similar(x, length(x)), max_batch)

function generateregions!(nextregions, a, b, sizes)
    ma = MVector(a)
    mb = MVector(b)
    n = length(a)
    Δ = map(/, b - a, sizes)

    @inbounds for c in CartesianIndices(sizes) # Val ntuple loops are unrolled
        for i = 1:n
            ma[i] = a[i]+(c[i]-1)*Δ[i]
            mb[i] = c[i]==sizes[i] ? b[i] : a[i]+c[i]*Δ[i]
        end
        x = oftype(a, ma)
        y = oftype(b, mb)

        push!(nextregions, (x, y))
    end
    return
end

defaultorder(::HAdaptError) = 4

function isconverged(criterion::HAdaptError, fun, nextfun::ChebPoly{n,T,Td}, val, order, norm, atol, rtol) where {n,T,Td}
    p = Iterators.product(ntuple(m -> range(nextfun.lb[m], nextfun.ub[m], length=order[m]*criterion.n), Val{n}())...)
    E = maximum(r_ -> (r = SVector{n,Td}(r_); norm(fun(r) - nextfun(r))), p)
    return E < max(atol, rtol*maximum(norm, val))
end

function nextchildrenregions!(nextregions, criterion::HAdaptError, fun::ChebPoly{n,T,Td}, val, order, norm, atol, rtol) where {n,T,Td}
    generateregions!(nextregions, fun.lb, fun.ub, ntuple(i->2,Val{n}()))
    return nothing
end

function defaultorder(criterion::SpectralError)
    order = 15
    @assert order > criterion.n "SpectralError requires that the order be greater than the number of Chebyshev coefficients to use for error estimation"
    return order
end

function isconvergeddims(criterion::SpectralError, fun::ChebPoly{n,T,Td}, val, order, norm, atol, rtol) where {n,T,Td}
    @assert minimum(order) > criterion.n "SpectralError requires that the order be greater than the number of Chebyshev coefficients to use for error estimation"
    tol = max(atol, rtol*maximum(norm, val))
    ntuple(Val{n}()) do i
        # assuming size(fun.coefs) = order
        idx = CartesianIndices(ntuple(j -> j==i ? (2+order[i]-criterion.n:size(fun.coefs,i)) : axes(fun.coefs, j), Val{n}()))
        tol > maximum(sum(norm, @view(fun.coefs[idx]); dims=i))
    end
end

function isconverged(criterion::SpectralError, parent, fun, val, order, norm, atol, rtol)
    return all(isconvergeddims(criterion, fun, val, order, norm, atol, rtol))
end

function nextchildrenregions!(nextregions, criterion::SpectralError, fun::ChebPoly{n,T,Td}, val, order, norm, atol, rtol) where {n,T,Td}
    dimsconverged = isconvergeddims(criterion, fun, val, order, norm, atol, rtol)
    all(dimsconverged) && return nothing

    newsize = map(v -> v ? 1 : 2, dimsconverged)
    generateregions!(nextregions, fun.lb, fun.ub, newsize)
    return nothing
end

# this function does batch evaluation while reusing already evaluated points
function batchevaluate!(x, f, t, order, valtree, funtree, searchtree, ninitdiv, reuse)
    vals = similar(first(valtree))
    reuse || return vals .= reshape(f(resize!(x, length(t)) .= vec(t)), size(vals))
    idx = typeof(first(CartesianIndices(t)))[]
    empty!(x)
    for i in CartesianIndices(t)
        ti = t[i]
        next = findevaluated(ti, CartesianIndices(axes(t)), valtree, funtree, searchtree, order, ninitdiv)
        if isnothing(next)
            push!(x, ti)
            push!(idx, i)
        else
            vals[i] = next
        end
    end
    vals[idx] .= f(x)
    return vals
end

function findevaluated(x, ind, valtree, funtree, searchtree, order, ninitdiv)
    tol = 10*eps(one(eltype(x)))
    treesearch(valtree, funtree, searchtree, x, ninitdiv) do fun, val, children
        next = findfirst(ind) do idx
            norm(_chebpoint(idx - oneunit(idx), order, fun.lb, fun.ub) - x) < tol
        end
        isnothing(next) ? nothing : val[ind[next]]
    end
end

function evalregions!!!!(callback, valtree, funtree, searchtree, t, nextregions, f, order, droptol, initdiv, reuse)
    #=
    # compile a list of all points to evaluate in the region
    for (x, y) in nextregions
        chebpoints!(t[i], order, x, y)
        # preallocate the output arrays
    end
    # remove redundancy in the current regions
    for (x, y) in nextregions
        if unique push!(queue) end
        # link nodes in region to items in queue
    end
    # remove redundancy from the parent regions to insert into the region values
    findevaluated

    =#
    for (x, y) in nextregions
        chebpoints!(t, order, x, y)
        nextval = batchevaluate!(f.x, f.f, t, order, valtree, funtree, searchtree, initdiv, reuse)
        nextfun = _chebinterp(nextval, x, y; tol=droptol)
        push!(valtree, nextval)
        push!(funtree, nextfun)
        push!(searchtree, Int[])
        callback(nextfun, nextval)
    end
end
evalregions!!!!(args...) = evalregions!!!!((_...) -> nothing, args...)

function hchebinterp_(criterion::AbstractAdaptCriterion, f::BatchFunction, a, b, order, atol_, rtol_, norm, maxevals, initdiv, droptol, reuse)
    maxevals < 0 && throw(ArgumentError("invalid negative maxevals"))

    nextregions = Tuple{typeof(a),typeof(a)}[]
    t = Array{typeof(a),length(a)}(undef, order .+ 1)

    ninitdiv = if initdiv isa TreePoly
        n = 0
        for (fun,children) in zip(initdiv.funtree, initdiv.searchtree)
            isempty(children) || continue
            push!(nextregions, (fun.lb, fun.ub))
            n += 1
        end
        n
    else
        generateregions!(nextregions, a, b, ntuple(i->initdiv, Val{length(a)}()))
        initdiv^length(a)
    end

    # unroll first iteration to get right types of buffers
    x0, y0 = popfirst!(nextregions)
    chebpoints!(t, order, x0, y0)
    resize!(f.x, length(t))
    f.x .= vec(t)
    valtree = [Array(reshape(f.f(f.x), size(t)))]
    funtree = [_chebinterp(valtree[1], x0, y0; tol=droptol)]
    searchtree = [Int[]]

    atol = something(atol_, zero(norm(first(valtree[1]))))
    rtol = something(rtol_, iszero(atol) ? sqrt(eps(one(atol))) : zero(one(atol)))

    evalregions!!!!(valtree, funtree, searchtree, t, nextregions, f, order, droptol, ninitdiv, reuse)
    empty!(nextregions)

    evalsperbox = prod(order)
    l = Ref(ninitdiv) # we pass a ref to avoid boxing in the inner closure

    queue = Int[]
    nextqueue = collect(1:ninitdiv)

    while !isempty(nextqueue)
        if l[]*evalsperbox > maxevals
            @warn "maxevals exceeded"
            break
        end

        copy!(queue, nextqueue)
        empty!(nextqueue)
        for i in queue
            val = valtree[i]
            fun = funtree[i]
            src = searchtree[i]

            nextchildrenregions!(nextregions, criterion, fun, val, order, norm, atol, rtol)
            evalregions!!!!(valtree, funtree, searchtree, t, nextregions, f, order, droptol, ninitdiv, reuse) do nextfun, nextval
                push!(src, l[] += 1)
                !isconverged(criterion, fun, nextfun, nextval, order, norm, atol, rtol) && push!(nextqueue, l[])
            end
            empty!(nextregions)
        end
    end
    return valtree, funtree, searchtree, ninitdiv
end

"""
    hchebinterp(f, a, b, [criterion=SpectralError()]; order=defaultorder(criterion), atol=0, rtol=0, norm=norm, maxevals=typemax(Int), initdiv=1, reuse=true)

Return a piecewise polynomial interpolant of `f` on the interval ``[a,b]`` of
degree `order` that is pointwise accurate to the requested tolerances. Uses
`criterion::AbstractAdaptCriterion` to estimate the interpolant error for
h-adaptation. By default, the `order` for `SpectralError()` is 15 and for
`HAdaptError()` is 4.


!!! note "HChebInterp 1.1"
    The `reuse` keyword requires at least HChebInterp v1.1.

!!! note "HChebInterp 1.2
    This version is required to pass a pre-evaluated interpolant as the `initdiv` keyword.

The keyword `reuse` specifies that the algorithm will reuse function evaluations
on the interpolation grid whenever possible. For expensive functions and
interpolation problems on the order of seconds, the benefit will be noticeable,
i.e. roughly a 12% saving in function evaluations for the default solver. Since
looking up the interpolation points is not necessarily fast, `reuse=false` can
be set to turn off this optimization.
"""
function hchebinterp(f::BatchFunction, a::SVector{n,T}, b::SVector{n,T};
    criterion=SpectralError(), order=defaultorder(criterion),
    atol=nothing, rtol=nothing, norm=norm, maxevals=typemax(Int),
    initdiv=1, droptol=zero(one(T)), reuse=true) where {n,T}
    @assert one(T) isa Real "a and b must be real vectors"
    s = Diagonal((b-a)/2)
    e = ones(SVector{n,typeof(one(T))})
    t2x = t -> a + s * (e + t)
    x = eltype(f.x) <: Nothing ? typeof(a)[] : f.x
    g = BatchFunction(typeof(e)[]) do t
        resize!(parent(x), length(t))
        f.f(map!(t2x, x, t))
    end
    ord = order isa Number ? ntuple(n->order, Val{length(a)}()) : promote(order...)
    valtree, funtree, searchtree, ninitdiv = hchebinterp_(criterion, g, -e, e, ord, atol, rtol, norm, maxevals, initdiv, droptol, reuse)
    return TreePoly(valtree, funtree, searchtree, a, b, ninitdiv)
end

function hchebinterp(f::F, a, b; kws...) where F
    (n = length(a)) == length(b) || throw(ArgumentError("a and b must be the same length"))
    z = float(zero(promote_type(eltype(a),eltype(b))))
    T = typeof(z)
    g = if a isa Number
        if f isa BatchFunction
            BatchFunction(x -> f.f(reinterpret(typeof(z), x)), eltype(f.x) <: Nothing ? f.x : reinterpret(SVector{n,typeof(z)}, f.x))
        else
            BatchFunction(x -> f.(reinterpret(typeof(z), x)))
        end
    else
        f isa BatchFunction ? f : BatchFunction(x -> f.(x))
    end
    return hchebinterp(g, SVector{n,T}(a), SVector{n,T}(b); kws...)
end

end
