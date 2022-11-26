begin
    using Random, LinearAlgebra, BenchmarkTools, Test, Statistics, StaticArrays
    using NearestNeighbors
    using StaticArrays
    using Flux3D
    using Rotations
    using ProfileCanvas
    using MeshIO,FileIO,GeometryBasics
end
include("utils.jl")
include("FRICP.jl")

begin
	PC, N = load_PC("./Assets/source.ply")
	Flux3D.normalize!(PC)
	source = PC2SVector(PC.points[:,:,1]);
	# mean = sum(source)/N
	# source = source .- [mean]
	Tree = KDTree(SVector2PC(source))
	R = rand(RotMatrix{3,Float32})
	t = @SMatrix rand(Float32,3,1)
	target = transform_PC(source,R,t)
end;

## Parameters in this cell

begin 
	function params(T)
		max_iter = 100;
		f = "welsch"
		aa = (νₛ = T(3.0), νₑ = T(1.0/(3.0*sqrt(3.0))), m = 5, d = 6, νₜₕ=T(1e-6),α=T(0.5))
		return (max_iter = max_iter, f = f, aa = aa, stop = T(1e-5))
	end;
	par = params(eltype(source[1]))
end

@time FICP_P2P(source,target,par)