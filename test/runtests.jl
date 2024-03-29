using Test
using HChebInterp

using Aqua

Aqua.test_all(HChebInterp)

@testset "HChebInterp" begin include("interp.jl") end
@testset "arrays" begin include("arrays.jl") end
@testset "batch" begin include("batch.jl") end
@testset "precision" begin include("precision.jl") end
@testset "units" begin include("units.jl") end
@testset "countevals" begin include("countevals.jl") end
@testset "initdiv" begin include("initdiv.jl") end
