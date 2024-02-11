using Test, HChebInterp

f = x -> imag(1/(x+1e-1im))

p, numa = hchebinterp_count(f, -1, 1)
_, numb = hchebinterp_count(f, -1, 1, initdiv=2)

@test numa > numb

g = x -> imag(1/(x+1e-2im))

p1, num1 = hchebinterp_count(g, -1, 1)

p2, num2 = hchebinterp_count(g, -1, 1, initdiv=p)
boxes = leafboxes(p)
p2b, num2b = hchebinterp_count(g, -1, 1, initdiv=boxes)

p3, num3 = hchebinterp_count(g, -1, 1, initdiv=p1)
boxes1 = leafboxes(p1)
p3b, num3b = hchebinterp_count(g, -1, 1, initdiv=boxes1)

xs = range(-1, 1, length=1013)
@test p1.(xs) == p2.(xs) == p3.(xs)
@test p2.(xs) == p2b.(xs)
@test num2 == num2b
@test p3.(xs) == p3b.(xs)
@test num3 == num3b
@test num1 > num2 > num3
