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

function hchebinterp_count_batch(f::BatchFunction, args...; kws...)
    ncall::Int = 0
    g = BatchFunction(f.x, f.max_batch) do x
        ncall += 1
        return f.f(x)
    end
    fun = hchebinterp(g, args...; kws...)
    return fun, ncall
end

f = x -> sin(exp(x^2))
b1 = BatchFunction(x -> f.(x); max_batch=1)
b20 = BatchFunction(x -> f.(x); max_batch=20)
b400 = BatchFunction(x -> f.(x); max_batch=400)
p1, n1 = hchebinterp_count_batch(b1, 0, 2; atol=1e-9)
p20, n20 = hchebinterp_count_batch(b20, 0, 2; atol=1e-9)
p400, n400 = hchebinterp_count_batch(b400, 0, 2; atol=1e-9)

xs = range(0, 2, length=2024)
@test f.(xs) ≈ p1.(xs) ≈ p20.(xs) ≈ p400.(xs)
@test n1 > n20 > n400 # check the number of batch calls decreases with larger max_batch
