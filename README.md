test

# HChebInterp.jl

| Documentation | Build Status | Coverage | Version |
| :-: | :-: | :-: | :-: |
| [![][docs-stable-img]][docs-stable-url] | [![][action-img]][action-url] | [![][codecov-img]][codecov-url] | [![ver-img]][ver-url] |
| [![][docs-dev-img]][docs-dev-url] | [![][pkgeval-img]][pkgeval-url] | [![][aqua-img]][aqua-url] | [![deps-img]][deps-url] |

This package provides multi-dimensional h-adaptive Chebyshev interpolation with the
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

The same routine also supports interpolation of multidimensional functions,
though note that the function must accept `SVector` inputs.
For example, a 2d function can be interpolated as follows:
```julia
g(x) = cis(prod(x))
p = hchebinterp(g, (0,0), (1,1); atol=1e-5)
```
with evaluation at points in the support of the interpolant like `p([0.1, 0.4])`.

The package can interpolate any numeric or array-valued types.
It also supports a batching interface.
See the documentation linked above for details.

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
- [ChebFun](https://github.com/chebfun/chebfun)
- [TreeFun](https://github.com/danfortunato/treefun)

<!-- badges -->

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://lxvm.github.io/HChebInterp.jl/stable/

[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://lxvm.github.io/HChebInterp.jl/dev/

[action-img]: https://github.com/lxvm/HChebInterp.jl/actions/workflows/CI.yml/badge.svg?branch=main
[action-url]: https://github.com/lxvm/HChebInterp.jl/actions/?query=workflow:CI

[pkgeval-img]: https://juliahub.com/docs/General/HChebInterp/stable/pkgeval.svg
[pkgeval-url]: https://juliahub.com/ui/Packages/General/HChebInterp

[codecov-img]: https://codecov.io/github/lxvm/HChebInterp.jl/branch/main/graph/badge.svg
[codecov-url]: https://app.codecov.io/github/lxvm/HChebInterp.jl

[aqua-img]: https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg
[aqua-url]: https://github.com/JuliaTesting/Aqua.jl

[ver-img]: https://juliahub.com/docs/HChebInterp/version.svg
[ver-url]: https://juliahub.com/ui/Packages/HChebInterp/UDEDl

[deps-img]: https://juliahub.com/docs/General/HChebInterp/stable/deps.svg
[deps-url]: https://juliahub.com/ui/Packages/General/HChebInterp?t=2
