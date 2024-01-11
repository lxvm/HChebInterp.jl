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

f = x -> imag(1/(x+1e-1im))

p, numa = hchebinterp_count(f, -1, 1)
_, numb = hchebinterp_count(f, -1, 1, initdiv=2)

@test numa > numb

g = x -> imag(1/(x+1e-2im))

p1, num1 = hchebinterp_count(g, -1, 1)

p2, num2 = hchebinterp_count(g, -1, 1, initdiv=p)

p3, num3 = hchebinterp_count(g, -1, 1, initdiv=p1)

@test num1 > num2 > num3
