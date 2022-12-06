begin
    using Random, LinearAlgebra, BenchmarkTools, Test, Statistics, StaticArrays
    using NearestNeighbors
    using StaticArrays
    using Rotations
    using ProfileCanvas
    using MeshIO,FileIO,GeometryBasics
end

include("../Scripts/utils.jl")
include("PC_Visualization.jl")

begin
	source, N = load_PC("./Assets/source.ply")
	target, Nₜ = load_PC("./Assets/target.ply")
    t2 = transform_PC(target,RotX{Float32}(-π/2)*RotZ{Float32}(π/2),@SMatrix zeros(Float32,3,1))
end;

visualize(source)
