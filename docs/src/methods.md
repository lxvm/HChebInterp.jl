# Manual

```@autodocs
Modules = [HChebInterp]
Private = false
Order   = [:type, :function]
```

```@docs
HChebInterp.AbstractAdaptCriterion
```

## Batching

We also support a batching interface with out-of-place functions that can be
parallelized by the user. For example

```julia
using HChebInterp

f = x -> cis(x^2)
b = BatchFunction(x -> f.(x))
p = hchebinterp(b, 0.0, 4.0)
```
