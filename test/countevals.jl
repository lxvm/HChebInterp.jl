# These tests check to see if HChebInterp reuses function evaluations that are
# coincidentally repeated due to the structure of the Chebyshev grid
using Test, HChebInterp

# interpolant requiring one subdivision, so 3 panels (HAdaptError also goes one
# subdivision further)
for ndim in 1:2
    for (criterion, numpanels) in ((HAdaptError(), 1+2^ndim+(2^ndim)^2), (SpectralError(), 1+2^ndim)),
        order in [4, 10]
        fun, numevals = hchebinterp_count(abs∘prod, -ones(ndim), ones(ndim), order=order, criterion=criterion)
        @test_broken numevals < numpanels*(order+1)^ndim
    end
end

# interpolant requiring two subdivisions, asymmetrically
for (criterion, numpanels) in ((HAdaptError(), 1+2+4+4), (SpectralError(), 1+2+2)), order in [4, 10]
    fun, numevals = hchebinterp_count(x -> min(1, abs(x)), -3, 1, order=order, criterion=criterion)
    @test_broken numevals <= numpanels*(order+1) - (numpanels-1)*3/2
    # in 1d we can save 1.5 evaluations per subdivided panel by reusing parent evaluations for children
    fun, numevals = hchebinterp_count(x -> min(1, abs(x)), -3, 1, order=order, criterion=criterion, reuse=false)
    @test numevals == numpanels*(order+1)
end
