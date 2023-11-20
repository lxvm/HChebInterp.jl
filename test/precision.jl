using Test, HChebInterp, StaticArrays

f1 = x -> x^3 - sin(x)
fun1 = hchebinterp(f1, Float32(0.0), Float32(1.0))
@test f1(Float32(0.5)) ≈ fun1(Float32(0.5))
@test typeof(f1(Float32(0.5))) == typeof(@inferred(fun1(Float32(0.5))))

f2 = x -> x*x'
fun2 = hchebinterp(f2, SVector(Float32(0.0),Float32(0.0)), SVector(Float32(1.0),Float32(1.0)))
@test f2(SVector(Float32(0.5),Float32(0.5))) ≈ fun2(SVector(Float32(0.5),Float32(0.5)))
@test typeof(f2(SVector(Float32(0.5),Float32(0.5)))) == typeof(@inferred(fun2(SVector(Float32(0.5),Float32(0.5)))))

# test precision of output
f3 = x -> ComplexF32(x'x)
fun3 = hchebinterp(f3, SVector(Float32(0.0),Float32(0.0)), SVector(Float32(1.0),Float32(1.0)))
@test f3(SVector(Float32(0.5),Float32(0.5))) ≈ fun3(SVector(Float32(0.5),Float32(0.5)))
@test typeof(f3(SVector(Float32(0.5),Float32(0.5)))) == typeof(@inferred(fun3(SVector(Float32(0.5),Float32(0.5)))))

# test that precision of output is maintained when precision of input changes
fun4 = hchebinterp(f3, SVector(0.0,0.0), SVector(1.0,1.0))
@test typeof(f3(SVector(0.5,0.5))) == typeof(@inferred(fun4(SVector(0.5,0.5))))
