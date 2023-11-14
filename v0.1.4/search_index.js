var documenterSearchIndex = {"docs":
[{"location":"methods/#Manual","page":"Manual","title":"Manual","text":"","category":"section"},{"location":"methods/","page":"Manual","title":"Manual","text":"Modules = [HChebInterp]\nPrivate = false\nOrder   = [:type, :function]","category":"page"},{"location":"methods/#HChebInterp.HAdaptError","page":"Manual","title":"HChebInterp.HAdaptError","text":"HAdaptError(; n=10)\n\nEstimate the error of the interpolant by dividing the panel into two, computing interpolants on the subpanels, and computing the maximum error between interpolants at n*p equispaced points, where p is the number of points used to compute each interpolant.\n\n\n\n\n\n","category":"type"},{"location":"methods/#HChebInterp.SpectralError","page":"Manual","title":"HChebInterp.SpectralError","text":"SpectralError(; n=3)\n\nEstimate the error of the interpolant by as the sum of the norm of the last n Chebyshev coefficients. Use abs to compute the norm of each coefficient.\n\n\n\n\n\n","category":"type"},{"location":"methods/#HChebInterp.hchebinterp-Tuple{Any, Any, Any}","page":"Manual","title":"HChebInterp.hchebinterp","text":"hchebinterp(f, a, b, [criterion=SpectralError()]; order=15, atol=0, rtol=0, norm=norm, maxevals=typemax(Int), initdiv=1)\n\nReturn a piecewise polynomial interpolant of f on the interval ab of degree order that is pointwise accurate to the requested tolerances. Uses criterion::AbstractAdaptCriterion to estimate the interpolant error for h-adaptation. If HAdaptError() is used as the criterion, it may be appropriate to reduce the order to 4 to avoid unnecessary function evaluations.\n\n\n\n\n\n","category":"method"},{"location":"methods/","page":"Manual","title":"Manual","text":"HChebInterp.AbstractAdaptCriterion","category":"page"},{"location":"methods/#HChebInterp.AbstractAdaptCriterion","page":"Manual","title":"HChebInterp.AbstractAdaptCriterion","text":"AbstractAdaptCriterion\n\nAbstract supertype for error criteria for adaptive refinement.\n\n\n\n\n\n","category":"type"},{"location":"#HChebInterp.jl","page":"Home","title":"HChebInterp.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"HChebInterp","category":"page"},{"location":"#HChebInterp","page":"Home","title":"HChebInterp","text":"A package for h-adaptive Chebyshev interpolation of N-D functions using FastChebInterp.jl. Algorithms based on work by Kaye et al.. See the tests for examples.\n\n\n\n\n\n","category":"module"}]
}
