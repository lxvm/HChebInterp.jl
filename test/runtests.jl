using Test
using HChebInterp

@testset "HChebInterp" begin
    for (criterion, order) in ((SpectralError(), 15), (HAdaptError(), 4))
        @testset "$(criterion)" begin
            # pick the unit interval for interpolation
            a, b = 0, 1
            # choose some functions to interpolate
            for f in (x -> sin(exp(x^2)), x -> exp(-sin(10x)^2))
                # construct interpolant
                p = hchebinterp(f, a, b, criterion)
                # check interpolation error is lower than default tolerance
                x = range(a, b, length=2000)
                @test f.(x) ≈ p.(x) atol = 1e-5
                # check out of bounds evaluation fails
                @test_throws ArgumentError p(2b)
            end
        end
    end
end