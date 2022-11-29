begin
	using MAT,Printf
    using Random, LinearAlgebra, BenchmarkTools, Test, Statistics, StaticArrays
    using NearestNeighbors
    using StaticArrays
    using Flux3D#master
    using Rotations
    using ProfileCanvas
    using MeshIO,FileIO,GeometryBasics
    using Makie,GLMakie
end

begin
	include("../Scripts/FRICP.jl")
	include("../Scripts/utils.jl")
end;

begin
	pc_path ="./Benchmark/bunny.mat";
	file = matopen(pc_path);src = read(file,"src");close(file);
	src = PC2SVector(src)
end;

begin
	Path = "/home/ssz990220/Project/JuliaPointCloud/Scan-matcher/data/random"
    outliers_per = 0; id = 1;
    pc_path = @sprintf("%s/%d/%d_dst.mat",Path,outliers_per,id)
	r_path = @sprintf("%s/%d/%d_R.mat",Path,outliers_per,id)
	file = matopen(pc_path)
	dst = read(file,"dst")
	T = eltype(dst)
	dst = PC2SVector(dst)
	close(file)
	file = matopen(r_path)
	R = SMatrix{3,3,T,9}(read(file,"R"))
	close(file)
	return dst,R
end;

begin 
	function params(T)
		max_iter = 100;
		f = "welsch"
		aa = (νₛ = T(3.0), νₑ = T(1.0/(3.0*sqrt(3.0))), m = 5, d = 6, νₜₕ=T(1e-6),α=T(0.5))
		return (max_iter = max_iter, f = f, aa = aa, stop = T(1e-5))
	end;
	par = params(eltype(src[1]))
end

T = FICP_P2P(src,dst,par)
transform_PC!(src,T)
Flux3D.normalize
begin
	fig = Figure(resolution=(600, 600))
	ax = Axis3(fig[1,1])
	visualize!(ax,PointCloud(SVector2PC(src)),color=:red)
	visualize!(ax,PointCloud(SVector2PC(dst)))
	fig
end
