using Test, LinearAlgebra
using HChebInterp

ninterp = 2000
for (criterion, order) in ((SpectralError(), 15), (HAdaptError(), 6))
    @testset "$(criterion)" begin
        # pick the unit interval for interpolation
        for n in 1:2, initdiv in [1, 3, 5]
            atol = rtol = 1e-3
            a, b = fill(0.0,n), fill(1.0,n)
            # choose some functions to interpolate
            for f in (
                x -> sin(exp(sum(abs2.(x)))),
                x -> exp(-sum(y ->sin(10y), x)^2),
                x -> 1e-1/(1e-2+(norm(x)-0.42)^2)*1e-1/(1e-2+(norm(x)-0.45)^2), # issue 10
            )
                # construct interpolant and specify atol
                p = hchebinterp(f, a, b; criterion=criterion, order=order, atol=atol, initdiv=initdiv)
                # check interpolation error is lower than default tolerance
                x = Iterators.product(ntuple(m -> range(a[m], b[m], length=ninterp), n)...)
                # check for pointwise (Linf) accuracy to atol
                @test maximum(y -> HChebInterp.norm(f(y) - p(y)), x) < atol

                # # construct interpolant and specify rtol
                p = hchebinterp(f, a, b; criterion=criterion, order=order, rtol=rtol)
                # check interpolation error is lower than default tolerance
                x = range(a, b, length=2000)
                # check for global (L2) accuracy to 2*rtol, since function decays
                @test maximum(y -> HChebInterp.norm(f(y) - p(y))/HChebInterp.norm(f(y)), x) < 2rtol

                # check out of bounds evaluation fails
                @test_throws ArgumentError p(2b)
            end
        end
    end
end
