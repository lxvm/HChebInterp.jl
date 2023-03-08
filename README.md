# HChebInterp.jl

[Documentation](https://lxvm.github.io/HChebInterp.jl/dev/)

This package provides 1-dimensional h-adaptive Chebyshev interpolation with the
routine `hchebinterp`. The routine uses a [Chebyshev
basis](https://en.wikipedia.org/wiki/Chebyshev_polynomials) to construct a
piecewise polynomial interpolant of an arbitrary smooth function. It depends on
[FastChebInterp.jl](https://github.com/stevengj/FastChebInterp.jl) and is
inspired by [Chebfun](https://www.chebfun.org/ ).

## Usage

To construct a polynomial interpolant `p` of the function `f` on the interval
`[a,b]` to a user-specified tolerance can be done with the interface
```julia
using HChebInterp
f(x) = sin(exp(x^2))
p = hchebinterp(f, 0, 1; atol=1e-5)
```
Then `p` can be evaluated using its functor interface at any point in the
interpolation interval, e.g `p(0.5)`.

## Algorithm

The algorithm of `hchebinterp` is based on the one described by [Kaye et
al., (2023)](http://arxiv.org/abs/2211.12959)
## Author and Copyright

HChebInterp.jl was written by [Lorenzo Van Muñoz](https://web.mit.edu/lxvm/www/),
and is free/open-source software under the MIT license.

## Related packages
- [FastChebInterp.jl](https://github.com/stevengj/FastChebInterp.jl)
- [ApproxFun.jl](https://github.com/JuliaApproximation/ApproxFun.jl)
- [Baobzi](https://github.com/flatironinstitute/baobzi)
- [TreeFun](https://github.com/danfortunato/treefun)