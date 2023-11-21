using Test, Unitful, HChebInterp, StaticArrays

f1 = x -> x^3 - sin(x)
uf1 = x -> f1(x/oneunit(x))
fun1 = hchebinterp(uf1, 0.0u"m", 1.0u"m")
@test uf1(0.5u"m") ≈ fun1(0.5u"m")
@test uf1(0.5u"m") ≈ fun1(5e5u"μm")

u2f1 = x -> u"s"*uf1(x)
fun2 = hchebinterp(u2f1, 0.0u"m", 1.0u"m")
@test u2f1(0.5u"m") ≈ fun2(0.5u"m")

umfn = x -> x*x'
mfn = x -> umfn(map(y->y/oneunit(y), x))
fun3 = hchebinterp(mfn, SVector(0.0,0.0)*u"m", SVector(1.0,1.0)*u"m")
@test mfn(SVector(0.5,0.5)*u"m") ≈ fun3(SVector(0.5,0.5)*u"m")
fun4 = hchebinterp(umfn, SVector(0.0,0.0)*u"m", SVector(1.0,1.0)*u"m")
@test umfn(SVector(0.5,0.5)*u"m") ≈ fun4(SVector(0.5,0.5)*u"m")
