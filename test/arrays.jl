# test interpolation of array-valued quantities
using Test, HChebInterp, StaticArrays

for f in (
    x -> fill(1.0),
    x -> ones(1),
    x -> ones(1,1),
    x -> Scalar(1.0),
    x -> ones(SVector{3,Float64}),
    x -> ones(SMatrix{3,3,Float64,9}),
    )
    fun = hchebinterp(f, 0.0, 1.0)
    @test fun(0.5) ≈ f(0.5)
end