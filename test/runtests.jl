using Test
using HChebInterp

using Aqua

Aqua.test_all(HChebInterp)

@testset "HChebInterp" include("interp.jl")
@testset "batch" include("batch.jl")
@testset "precision" include("precision.jl")
@testset "units" include("units.jl")
