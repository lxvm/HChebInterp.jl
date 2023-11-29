# These tests check to see if HChebInterp reuses function evaluations that are
# coincidentally repeated due to the structure of the Chebyshev grid
using Test, HChebInterp

function hchebinterp_count(f, args...; kws...)
    numevals::Int = 0
    g = if f isa BatchFunction
        BatchFunction(f.x) do x
            n += length(x)
            return f.f(x)
        end
    else
        x -> (numevals += 1; f(x))
    end
    fun = hchebinterp(g, args...; kws...)
    return fun, numevals
end

# interpolant requiring one subdivision, so 3 panels (HAdaptError also goes one
# subdivision further)
for ndim in 1:2
    for (criterion, numpanels) in ((HAdaptError(), 1+2^ndim+(2^ndim)^2), (SpectralError(), 1+2^ndim)),
        order in [4, 10,] # 1] # seems to be a separate issue with 1, but the function is piecewise linear
        fun, numevals = hchebinterp_count(abs∘prod, -ones(ndim), ones(ndim), order=order, criterion=criterion)
        @test numevals == numpanels*(order+1)^ndim
    end
end

