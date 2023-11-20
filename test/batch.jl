using Test, HChebInterp

f = x -> cis(x^2)
b = BatchFunction(x -> f.(x))
pf = hchebinterp(f, 0.0, 4.0)
pb = hchebinterp(b, 0.0, 4.0)


@test pf(1pi) == pb(1pi)

order = 19
b2 = BatchFunction(x -> f.(x), Vector{Float64}(undef, order+1))
pb2 = hchebinterp(b2, 0.0, 4.0, order=order)

@test pf(1pi) ≈ pb2(1pi)
