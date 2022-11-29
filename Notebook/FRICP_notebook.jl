### A Pluto.jl notebook ###
# v0.19.16

using Markdown
using InteractiveUtils

# ╔═╡ e4532490-63e8-11ed-2f97-632853f48f1d
begin
	using Random, LinearAlgebra, BenchmarkTools, Test, Statistics, StaticArrays
	using NearestNeighbors
	using StaticArrays
	using Flux3D
	using Rotations
	using ProfileCanvas
	using MeshIO,FileIO,GeometryBasics
	using MAT,Printf
	using PlutoUI
	using StatsBase
end

# ╔═╡ 5261d83a-b73f-4a7f-a101-5c57b47d29d1
using GLMakie, Makie

# ╔═╡ 5436358a-86ed-4724-a4b1-58ed3cb18d32
md"# Fast and Robust ICP
*Juyong Zhang and Yuxin Yao and Bailin Deng*, 
*IEEE Transactions on Pattern Analysis and Machine Intelligence*"

# ╔═╡ 012ef84b-48cb-4932-9879-48acc2e1f39d
md"## Load a Point Cloud"

# ╔═╡ ab5a6a07-d2b5-43fa-9ca2-98ea2c9438ad
# ╠═╡ disabled = true
#=╠═╡
begin
	pc_path ="../Benchmark/bunny.mat";
	file = matopen(pc_path);src = read(file,"src");close(file);
	src = PC2SVector(src)
end;
  ╠═╡ =#

# ╔═╡ 8f05aef0-903e-4c20-8fad-3ee03e43dda2
# ╠═╡ disabled = true
#=╠═╡
begin
	Path = "/home/ssz990220/Project/JuliaPointCloud/Scan-matcher/data/random"
	outliers_pers = [0,20,60,70,80,90,95,96,97,98,99]
	id = collect(1:40)
	local iterator = collect(Iterators.product(outliers_pers,id))
	i = 5
	outliers_per = iterator[i][1]; id = iterator[i][2]
	dst,R = load_PC_Mat(Path,outliers_per,id)
end
  ╠═╡ =#

# ╔═╡ 191c1a65-d627-4595-88df-d5b5c73edcdf
md"## Parameters"

# ╔═╡ c462804d-5ea6-4fb7-baf9-861c9c961fe7
md"## FRICP"

# ╔═╡ aba902a4-7dd2-42cc-84bc-62c6c5957e4b
# begin
# 	q,w = get_pc_closest_point(s,t,KDTree(SVector2PC(t)))
# 	Transform = P2P_ICP(s,q)
# end

# ╔═╡ 3c520c56-d555-4749-92ff-28ea3b541160
#=╠═╡
R
  ╠═╡ =#

# ╔═╡ 11aebf61-cf36-450f-aa36-af3508844553
md"## Structs"

# ╔═╡ f39ff907-bc7c-49bd-b813-65ad03f4b190
md"## Functions"

# ╔═╡ bf054703-1880-4b01-8cd8-35fcf7c37973
@inline SRange(n) = StaticArrays.SUnitRange(n,n)

# ╔═╡ 521daff7-e4dc-43b3-aa7c-3e543c5b6ffe
function AA_Acc!(G::SMatrix{D,1,T,D},Fs,Gs,U,scales,iter,par) where {T,D}
	F = G .- U
	if iter == 1
		Fs[:,1] = -F
		Gs[:,1] = -G
		return G
	else
		m = par.aa.m
		col = mod(iter-1,m) == 0 ? m : mod(iter-1,m)
		theta = @MMatrix zeros(T,1,m)
		
		Fs[:,SRange(col)] .+= F
		Gs[:,SRange(col)] .+= G
		scale = maximum([T(1e-14),norm(Fs[:,SRange(col)])])
		scales[:,SRange(col)] = scale
		Fs[:,SRange(col)] .= Fs[:,SRange(col)] ./ scale
	
		# Incremental Update and solve
		mₖ = minimum([par.aa.m,iter-1])
		if mₖ == 1
			Fn = norm(Fs[:,SRange(col)])^2; Fnᵣ = sqrt(Fn);
			if (Fnᵣ > 1e-14)		## Prevent Zero Division
				theta[1] = (transpose(Fs[:,SRange(col)]./Fnᵣ)*(F./Fnᵣ))[1]
				## Triangle Projection
			end
		else
			## Solve Linear Least Square for AA
			theta[SOneTo(mₖ)] = Fs[:,SOneTo(mₖ)]\F
		end
	
		## Assemble the acc result
		U .= G .- Gs[:,SOneTo(mₖ)]*(theta[:,SOneTo(mₖ)]./scales[:,SOneTo(mₖ)])'
		## Prepare for next iter
		col += 1
		col = mod(col,m) == 0 ? m : mod(col,m)
		Fs[:,SRange(col)] .= -F
		Gs[:,SRange(col)] .= -G
	end
	return U	
end

# ╔═╡ 77986ac9-66ed-46b1-9a2f-e9a7dfa812d2
"""
    PC2SVector(PC)
Convert a point cloud to array of StaticMatrices
"""
function PC2SVector(PC::Matrix{T}) where {T<:Number}
    n = size(PC,2);
    PCVec = Array{SMatrix{3,1,T,3}}(undef,n)
    Threads.@threads for i = 1:n
        @inbounds PCVec[i] = @SMatrix [PC[1,i];PC[2,i];PC[3,i]]
    end
    return PCVec
end

# ╔═╡ fa55bd44-d803-41f8-a48e-f7bc03b66dec
function load_PC_Mat(Path,outliers_per,id)
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
end

# ╔═╡ 5674df53-78ab-404e-859b-7b8abdaad2b3
function Point2PC(P::Vector{GeometryBasics.Point{D,T}}) where {D,T}
	n = size(P,1)
	PC = zeros(T,D,n)
	for i = 1:n
		PC[:,i] = P[i]
	end
	return PC
end

# ╔═╡ 1eee9776-8495-45dc-86dc-b05c16bea058
"""
    load_PC(Path)
Load a point cloud from given Path. Returns a point cloud `P` and # of Points `n`
"""
function load_PC(Path)
    P = load(Path)
	PC = P.position
	P = Point2PC(PC)
    P = PointCloud(P)
    n = size(P.points,2)
    return P, n
end

# ╔═╡ ec83c06c-7480-4b27-8f42-b0794330657a
"""
	Svector2PC(PC)
Convert vector of Points in SMatrix to a Matrix
"""
function SVector2PC(P::Vector{SMatrix{D,1,T,D}}) where {T<:Number,D}
	n = size(P,1);
	PC = zeros(T,D,n)
	Threads.@threads for i = 1:n
		@inbounds PC[:,i] = P[i][:]'
	end
	return PC
end

# ╔═╡ 0388e7b0-95d7-46b9-9a37-00180052d6dc
"""
	get_pc_closest_point(X,Y,Tree)
Find the index & point coordiante of the closest point for each point in `X::Vector{SMatrix}` from KDTree `Tree::KDTree`, which is built from point cloud `Y`. The indices are filled in `W`. Points are assigned in `Q`
"""
function get_pc_closest_point(X::Vector{SMatrix{D,1,T,D}},Y::Vector{SMatrix{D,1,T,D}},Tree) where{T,D}
	n = size(X,1)
	Q = Array{SMatrix{D,1,T,D}}(undef,n)
	W = zeros(T,n)
	Threads.@threads for i = 1:n
		@inbounds idx,dist = nn(Tree,X[i])
		@inbounds W[i] = dist[1]
		@inbounds Q[i] = Y[idx[1]]
	end
	return Q,W
end

# ╔═╡ ab7d59c1-e7fa-45f9-bb84-0af2bf8b8fce
"""
	get_pc_closest_point!(X,Y,Tree,Q,W)
Find the index & point coordiante of the closest point for each point in `X::Vector{SMatrix}` from KDTree `Tree::KDTree`, which is built from point cloud `Y`. The indices are filled in `W`. Points are assigned in `Q`
"""
function get_pc_closest_point!(X::Vector{SMatrix{D,1,T,D}},Y::Vector{SMatrix{D,1,T,D}},Tree,Q::Vector{SMatrix{D,1,T,D}},W::Vector{T}) where{T,D}
	@assert size(X) == size(Q)
	@inbounds Threads.@threads for i ∈ eachindex(X)
		idx,dist = nn(Tree,X[i])
		W[i] = dist[1]
		Q[i] = Y[idx[1]]
	end
	return Q,W
end

# ╔═╡ bc29f7cd-5f96-491f-ac03-dd0f01f574ae
"""
	get_pc_closest_point_id(PC,Tree,W)
Find the index of the closest point for each point in `P::Vector{SMatrix}` from KDTree `Tree::KDTree`. The indices are filled in `W`
"""
function get_pc_closest_point_id(P::Vector{SMatrix{3,1,T,3}},Tree) where{T}
	n = size(P,1)
	id = zeros(n)
	Threads.@threads for i = 1:n
		@inbounds idx = nn(Tree,P[i])[1][1]
		@inbounds id[i] = idx
	end
	return id
end

# ╔═╡ 8a7ce52a-403f-45e7-b9b8-b4b5b46c69ac
"""
	transform_PC!(PC,R,t)
Inplace transformation of a Point Cloud `PC` under rotation `R` and translation `t`

See also [`transform_PC`](@ref)
"""
function transform_PC!(PC::Vector{SMatrix{3,1,T,3}}, R::T₂, t::T₃) where {T<:Number,T₂<:AbstractMatrix{T},T₃<:AbstractMatrix{T}}
	Threads.@threads for i ∈ eachindex(PC)
		@inbounds PC[i] = R*PC[i] .+ t;
	end
end

# ╔═╡ 01a9c86d-f0b2-4a6c-ae07-0f7c0090ca91
function normalize_PC(PC::Vector{SMatrix{D,1,T,D}}) where {T,D}
	n = size(PC,1);
	mean = sum(PC)/n;
	PC_ = PC .- [mean];
	scale = maximum(norm.(PC_));
	PC_ = PC_ ./ scale;
	return PC_, scale
end

# ╔═╡ bfd6c10c-04da-4e52-a0f2-1641e4572317
function normalize_PC!(PC::Vector{SMatrix{D,1,T,D}}) where {T,D}
	n = size(PC,1);
	mean = sum(PC)/n;
	PC .= PC .- [mean];
	scale = maximum(norm.(PC));
	PC .= PC ./ scale;
	return scale
end

# ╔═╡ b0ef0120-4385-457f-8104-217de22ba4fa
begin
	PC, N = load_PC("../Assets/source.ply")
	s = PC2SVector(PC.points[:,:,1]);
	scaleᵢ = normalize_PC!(s)
	PCₜ, Nₜ = load_PC("../Assets/target.ply")
	t = PC2SVector(PCₜ.points[:,:,1])
	scaleₜ = normalize_PC!(t)
	# # scaleₜ = normalize_PC!(t)
	# Nₜ = size(t,1)
	shuffle_idx = sample(1:Nₜ, Nₜ, replace = false)
	t = t[shuffle_idx]
end;

# ╔═╡ 654f8ec6-ee3a-4570-b122-03cab1955c47
begin 
	function params(T)
		max_iter = 100;
		f = "welsch"
		aa = (νₛ = T(3.0), νₑ = T(1.0/(3.0*sqrt(3.0))), m = 5, d = 6, νₜₕ=T(1e-6),α=T(0.5))
		return (max_iter = max_iter, f = f, aa = aa, stop = T(1e-5))
	end;
	par = params(eltype(s[1]))
end

# ╔═╡ 65e95d65-c0ec-4568-a52e-3b9242f68494
"""
	rt2T(R,t)
Combine a rotation matrix `R` and a translation vector `t` to a homogenous matrix `T`
"""
function rt2T(R::T₁,t::SMatrix{D,1,T,D}) where {D,T₁<:AbstractMatrix,T<:Number}
	row = zeros(1,D+1); row[1,D+1] = T(1.0);
	row = SMatrix{1,D+1,T,D+1}(row);
	return vcat(hcat(R,t),row)
end

# ╔═╡ fbdbf4fd-bd9e-4844-890a-a7731279089d
"""
	P2P_ICP(source,target, w)
Point to Point ICP with SVD. **Points in source and points in target are listed correspoindingly, given weight `w`**
"""
function P2P_ICP(X::Vector{SMatrix{D,1,T,D}},Y::Vector{SMatrix{D,1,T,D}},w::Vector{T}=ones(T,size(X,1))) where {T<:Number,D}
	wₙ = w/sum(w);
	meanₛ = sum(X.*wₙ);X = X.-[meanₛ];
	meanₜ = sum(Y.*wₙ);Y = Y.-[meanₜ];
	Σ = reduce(hcat, X) * reduce(hcat, Y.*wₙ)'
	F = svd(Σ);
	U = SMatrix{D,D,T,D*D}(F.U);
	V = SMatrix{D,D,T,D*D}(F.V);
	if det(U)*det(V) < 0
		s = ones(T,D); s[end] = -s[end];
		S = SMatrix{D,D,T,D*D}(diagm(s))
		R = V*S*U';
	else
		R = V*U';
	end
	t = meanₜ-R*meanₛ
	X = X.+ [meanₛ]; Y = Y.+ [meanₜ];
	return rt2T(R,t)
end

# ╔═╡ d1886891-e260-4104-866a-ead8284af0ce
"""
	T2rt(T)
Decompose a homogenous transformation matrix `T` into a rotation matrix `R` and a translational vector `t`
"""
function T2rt(T)
	R = T[SOneTo(3),SOneTo(3)]
	t = T[SOneTo(3),4]
	return R,t
end

# ╔═╡ 1e902be5-c98d-421a-8ef4-7294e6855640
begin
"""
transform_PC(PC,R,t)
Transforming a Point Cloud `PC` under rotation `R` and translation `t`

See also [`transform_PC!`](@ref)
"""
	function transform_PC(PC::Vector{SMatrix{3,1,T,3}}, R::T₂, t::T₃) where {T<:Number,T₂<:AbstractMatrix{T},T₃<:AbstractMatrix{T}}
		n = size(PC,1)
		PC_ = Array{SMatrix{3,1,T,3}}(undef,n)
		Threads.@threads for i ∈ eachindex(PC)
			@inbounds PC_[i] = R*PC[i] .+ t;
		end
		return PC_
	end
	function transform_PC(PC::Vector{SMatrix{3,1,T₁,3}}, T::T₂) where {T₁<:Number,T₂<:AbstractMatrix{T₁}}
		n = size(PC,1)
		R,t = T2rt(T)
		PC_ = Array{SMatrix{3,1,T₁,3}}(undef,n)
		Threads.@threads for i ∈ eachindex(PC)
			@inbounds PC_[i] = R*PC[i] .+ t;
		end
		return PC_
	end
end

# ╔═╡ 56844bba-6900-4c19-8925-03d5ae307599
"""
	transform_PC!(PC,T)
Inplace transformation of a Point Cloud `PC` under transformation `T`

See also [`transform_PC`](@ref)
"""
function transform_PC!(PC::Vector{SMatrix{3,1,T₁,3}}, T::T₂) where {T₁<:Number,T₂<:AbstractMatrix{T₁}}
	R,t = T2rt(T)
	Threads.@threads for i ∈ eachindex(PC)
		@inbounds PC[i] = R*PC[i] .+ t;
	end
end

# ╔═╡ 26fece0f-2e3c-4966-81fa-ce794b2079c6
begin
	function tukey_energy(r::Vector{T},p::T) where {T}
		r = (1 .-(r./p).^2).^2;
		r[r.>p] = T(0.0);
		return r
	end
	function trimmed_energy(r::Vector{T},p::T) where {T}
		return zeros(T,size(r,1))
		## TODO: finish trimmed_energy function
	end
	fair_energy(r,p) = @inline sum(r.^2) ./ (1 .+r./p);
	logistic_energy(r,p) = @inline sum(r.^2 .*(p./r)*tanh.(r./p));
	welsch_energy(r::Vector{T},p::T) where {T} = @inline sum(T(1.0).-exp.(-r.^2 ./(T(2.0) .*p.*p)));
	autowelsch_energy(r::Vector{T},p::T) where {T} = welsch_energy(r,T(0.5));
	uniform_energy(r::Vector{T}) where {T} = @inline ones(T,size(r,1))
	md"Energy functions here"
end

# ╔═╡ 53f8e4b5-d39b-44a6-b8d7-017a94883293
"""
	get_energy(W,ν₁,f)
get point cloud error energy given error `W`, parameter `ν₁`, with function specified by `f`
"""
function get_energy(W,ν₁,f)
	if f == "tukey"
		return tukey_energy(W,ν₁)
	elseif f == "fair"
		return fair_energy(W,ν₁)
	elseif f == "log"
		return logistic_energy(W,ν₁)
	elseif f == "trimmed"
		return trimmed_energy(W,ν₁)
	elseif f == "welsch"
		return welsch_energy(W,ν₁)
	elseif f == "auto_welsch"
		return autowelsch_energy(W,ν₁)
	elseif f == "uniform"
		return uniform_energy(W)
	else
		return uniform_energy(W)
	end
end

# ╔═╡ 433792e0-0877-4ac8-971a-978e4fcf60bd
"""
	median(v)
Find the median value of a vector `v`. `v` should be sorted ahead.
"""
function median(v::Vector{T}) where {T}
	n = size(v,1)
	if iseven(n)
		return (v[n÷2] + v[n÷2+1])/T(2.0) 
	else
		return (v[(n+1)÷2] + v[(n-1)÷2])/T(2.0)
	end
end

# ╔═╡ b7bb98d7-9469-4611-ab27-ddca18b9cfb5
begin
	uniform_weight(r::Vector{T}) where {T} = @inline ones(T,size(r,1))
	pnorm_weight(r::Vector{T},p::T,reg=T(1e-16)) where {T} = @inline p./(r.^(2-p) .+ reg)
	tukey_weight(r::Vector{T},p::T) where {T} = @inline (T(1.0) .- (r ./ p).^(T(2.0))).^T(2.0)
	fair_weight(r::Vector{T},p::T) where {T} = @inline T(1.0) ./ (T(1.0) .+ r ./ p)
	logistic_weight(r::Vector{T},p::T) where {T} = @inline (p ./ r) .* tanh.(r./p)
	welsch_weight(r::Vector{T},p::T) where {T} = @inline exp.(-(r.*r)./(2*p*p))
	autowelsch_weight(r::Vector{T},p::T) where {T} = welsch_weight(r,p*median(r)/T(sqrt(2.0)*2.3))
	function trimmed_weight(r::Vector{T},p::T) where {T}
		return error
	end
	md"weight functions here"
end

# ╔═╡ 22dd669d-f9ec-4b54-b161-7a4ffa8ef708
"""
	robust_weight!(W,ν₁,f)
get point cloud robust weight given error `W`, parameter `ν₁`, with function specified by `f`
"""
function robust_weight!(W,ν₁,f)
	if f == "tukey"
		W.= tukey_weight(W,ν₁)
	elseif f == "fair"
		W.=  fair_weight(W,ν₁)
	elseif f == "log"
		W.=  logistic_weight(W,ν₁)
	elseif f == "trimmed"
		W.=  trimmed_weight(W,ν₁)
	elseif f == "welsch"
		W.=  welsch_weight(W,ν₁)
	elseif f == "auto_welsch"
		W.=  autowelsch_weight(W,ν₁)
	elseif f == "uniform"
		W.=  uniform_weight(W)
	else
		W.=  uniform_weight(W)
	end
end

# ╔═╡ 33020dfe-eaef-47b6-800f-8329109de36b
function FindKnearestMed(P::Vector{SMatrix{3,1,T,3}},Tree,k) where {T}
	n = size(P,1)
	Xnearest = Vector{T}(undef,n)
	Threads.@threads for i = 1:n
		idxs, dists = knn(Tree, P[i], k, true)
		Xnearest[i] = median(dists[1])
	end
	return sqrt(median(Xnearest))
end

# ╔═╡ 16e90931-721e-4dbf-b921-b68f119a4476
function MatrixLog3(R::AbstractMatrix{T}) where {T<:Number}
	acosinput = (tr(R) - T(1.0)) / T(2.0);
	if acosinput >= 1
		so3mat = zeros(T,3,3);
	elseif acosinput <= -1
		if ~NearZero(T(1.0) + R[3, 3])
			omg = (T(1.0) / sqrt(T(2.0) * (T(1.0) + R[3, 3])))* [R[1,3]; R[2,3]; T(1.0) + R[3,3]];
		elseif ~NearZero(T(1.0) + R[2, 2])
			omg = (T(1.0) / sqrt(T(2.0) * (T(1.0) + R[2, 2])))* [R[1, 2]; T(1.0) + R[2, 2]; R[3, 2]];
		else
			omg = (T(1.0) / sqrt(T(2.0) * (T(1.0) + R[1, 1])))* [T(1.0) + R[1, 1]; R[2, 1]; R[3, 1]];
		end
		so3mat = mcross(pi * omg);
	else
		theta = acos(acosinput);
		so3mat = theta * (T(1.0) / (T(2.0) * sin(theta))) * (R - R');
	end
	return so3mat
end

# ╔═╡ 9eba2bdd-1ccf-4621-8df0-9ed3eec8707a
function wrapto1(a::T) where {T<:Number}
	if a > 1 
		return T(1.0)
	elseif a < -1
		return T(-1.0)
	else
		return a
	end
end

# ╔═╡ 7b50b05c-b3bf-468c-95da-7ddcf72479a0
function MatrixLog6(T::AbstractMatrix{D}) where {D<:Number}
	R, p = T2rt(T);
	omgmat = MatrixLog3(R);
	if isequal(omgmat, zeros(D,3,3))
		R = @SMatrix zeros(D,3,3)
		expmat = vcat(hcat(R,T[SOneTo(3), 4]) ,@SMatrix zeros(D,1,4));
	else
		theta = acos(wrapto1((tr(R) - 1.0f0) / 2.0f0));
		expmat = vcat(hcat(omgmat,(I - omgmat ./ 2.0f0 + (1.0f0 / theta - cot(theta / 2.0f0) / 2.0f0) * omgmat * omgmat / theta) * p),@SMatrix zeros(D,1,4));
	end
	return expmat
end

# ╔═╡ 9f17db92-7724-4459-bc50-1a8fa27944ac
se32vec(se3) = @SMatrix [-se3[2,3];se3[1,3];-se3[1,2];se3[1,4];se3[2,4];se3[3,4]]

# ╔═╡ 832d9d3d-5379-49a9-8a82-6486a835573e
vec2se3(vec::StaticArray{Tuple{D, 1}, T, 2}) where {T,D}  = @SMatrix [T(0.0) -vec[3] vec[2] vec[4]; 
										vec[3] T(0.0) -vec[1] vec[5];
										-vec[2] vec[1] T(0.0) vec[6]; 
										T(0.0) T(0.0) T(0.0) T(0.0)]

# ╔═╡ 2350385e-81e0-47ce-b13d-0cbbbfed4894
function FICP_P2P(source::Vector{SMatrix{D,1,T,D}},target::Vector{SMatrix{D,1,T,D}},param) where {T<:Number,D}
	## Setup Buffer
	n = size(source,1)
	X = deepcopy(source)
	Y = target; Tree = KDTree(SVector2PC(Y));
	if haskey(par,:init)
		Tₘ = par.init ? par.T : MMatrix{D+1,D+1,T,(D+1)^2}(I)
	else
		Tₘ = MMatrix{D+1,D+1,T,(D+1)^2}(I)
	end
	local energys = zeros(T,par.max_iter)
	local last_energy = typemax(T)
	local T_p2p = @SMatrix zeros(T,D+1,D+1)
	local Tₗₐₛₜ = @MMatrix zeros(T,D+1,D+1)
	local Q = Vector{SMatrix{D,1,T,D}}(undef,n)
	local W = zeros(T,n)
	
	## Initial Closest Point and Weights
	get_pc_closest_point!(X,Y,Tree,Q,W)
	
	## Welsch Parameters
	ν₂ = par.aa.νₑ * FindKnearestMed(target,Tree,7)
	ν₁ = par.aa.νₛ * median(W)
	ν₁ = ν₁ > ν₂ ? ν₁ : ν₂
	@show ν₁,ν₂

	## AA Buffers
	d = D == 3 ? 6 : 4 		# 3D & 2D only
	scales = @MMatrix zeros(T,1,par.aa.m)
	local U = @MMatrix zeros(T,d,1)
	local Tᵥ = @MMatrix zeros(T,d,1)
	local Gs = @MMatrix zeros(T,d,par.aa.m,)
	local Fs = @MMatrix zeros(T,d,par.aa.m,)
	local iter = 0
	local counter = 0
	local outer_loop = 0

	## Main Loop
	while true
		iter = 1
		for i = 1:par.max_iter
			acceptₐₐ = false
			energy = get_energy(W,ν₁,"welsch")
			if energy < last_energy
				last_energy = energy
				acceptₐₐ = true
			else
				U .= se32vec(MatrixLog6(T_p2p))
				X_ = transform_PC(X,T_p2p)
				get_pc_closest_point!(X_,Y,Tree,Q,W)
				last_energy = get_energy(W,ν₁,"welsch")
			end
			energys[i] = last_energy
			robust_weight!(W,ν₁,"welsch")
			T_p2p = P2P_ICP(X,Q,W)
			Tᵥ .= AA_Acc!(se32vec(MatrixLog6(T_p2p)),Fs,Gs,U,scales,iter,par)
			Tₘ = exp(vec2se3(Tᵥ))
			
			X_ = transform_PC(X,Tₘ)
			get_pc_closest_point!(X_,Y,Tree,Q,W)
			iter += 1
			counter += 1
			## Global Stop Criteria
			stop = norm(Tₘ-Tₗₐₛₜ)
			Tₗₐₛₜ .= Tₘ
			if stop<par.stop
				break
			end
		end
		if abs(ν₁-ν₂) < par.aa.νₜₕ
			break
		else
			ν₁ = ν₁*par.aa.α > ν₂ ? ν₁*par.aa.α : ν₂
			last_energy = typemax(T)
		end
		outer_loop += 1
		@show outer_loop,counter
		counter = 0
	end 
	return Tₘ
end

# ╔═╡ c7fdedf4-0ece-4cc8-8752-fbfd78696d09
Transform = FICP_P2P(s,t,par)
# Transform = P2P_ICP(s,t)

# ╔═╡ 0f8abea4-3fdc-4b80-9a98-c4b054433b18
begin
	"""
	    visualize(pcloud::PointCloud; kwargs...)
	
	Visualize PointCloud `pcloud` at `index`.
	
	Dimension of points in PointCloud `pcloud` must be 3.
	
	### Optional Arguments:
	- color (Symbol)       - Color of the marker, default `:blue`
	- markersize (Number)  - Size of the marker, default `npoints(pcloud)/5000`
	
	"""
	function visualize(p::PointCloud, index::Number = 1; kwargs...)
	    points = cpu(p[index])
	    size(points, 1) == 3 || error("dimension of points in PointCloud must be 3.")
	
	    kwargs = convert(Dict{Symbol,Any}, kwargs)
	    get!(kwargs, :color, :lightgreen)
	    get!(kwargs, :markersize, npoints(p) / 10000)
	
	    Makie.meshscatter(points[1, :], points[2, :], points[3, :]; kwargs...)
	end
	
	function visualize!(axis3::Makie.Axis3, p::PointCloud, index::Number = 1; kwargs...)
	    points = cpu(p[index])
	    size(points, 1) == 3 || error("dimension of points in PointCloud must be 3.")
	
	    kwargs = convert(Dict{Symbol,Any}, kwargs)
	    get!(kwargs, :color, :lightgreen)
	    get!(kwargs, :markersize, npoints(p) / 10000)
	
	    Makie.meshscatter!(axis3, points[1, :], points[2, :], points[3, :]; kwargs...)
	end
end

# ╔═╡ 1d1a3f08-ede0-4cea-8aea-a5aee6aaeaed
md"## Plots"

# ╔═╡ fa1fd339-2afe-467c-90f7-02bc0410c7d7
begin	
	fig = Figure(resolution=(600, 600))
	empty!(fig)
end

# ╔═╡ 737d6393-a015-4037-bc91-ff93e25fcdf5
fig

# ╔═╡ bfac69ec-9f99-46f8-b379-15ff98f8799d
ax = Axis3(fig[1,1])

# ╔═╡ 6b7645ec-03e2-4eec-83bd-bac1650c5c1a
visualize!(ax,PointCloud(SVector2PC(transform_PC(s,Transform))),markersize=0.005)
# pl = visualize!(ax,PointCloud(SVector2PC(s)),markersize=0.005)

# ╔═╡ 3c852557-bdfe-4c64-85b5-c8ef84333238
visualize!(ax,PointCloud(SVector2PC(t)),color=:red,markersize=0.005)

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
FileIO = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
Flux3D = "432009dd-59a1-4b72-8c93-6462ce9b220f"
GLMakie = "e9467ef8-e4e7-5192-8a1a-b1aee30e663a"
GeometryBasics = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
MAT = "23992714-dd62-5051-b70f-ba57cb901cac"
Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
MeshIO = "7269a6da-0436-5bbc-96c2-40638cbb6118"
NearestNeighbors = "b8a86587-4115-5ab1-83bc-aa920d37bbce"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Printf = "de0858da-6303-5e67-8744-51eddeeeb8d7"
ProfileCanvas = "efd6af41-a80b-495e-886c-e51b0c7d77a3"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Rotations = "6038ab10-8711-5258-84ad-4b1120ba62dc"
StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[compat]
BenchmarkTools = "~1.3.2"
FileIO = "~1.16.0"
Flux3D = "~0.1.6"
GLMakie = "~0.7.3"
GeometryBasics = "~0.4.5"
MAT = "~0.10.3"
Makie = "~0.18.3"
MeshIO = "~0.4.10"
NearestNeighbors = "~0.4.12"
PlutoUI = "~0.7.48"
ProfileCanvas = "~0.1.6"
Rotations = "~1.3.3"
StaticArrays = "~1.5.9"
StatsBase = "~0.33.21"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.8.2"
manifest_format = "2.0"
project_hash = "617592fa0d941aa2793e233e93a2e8358c7959ce"

[[deps.AbstractFFTs]]
deps = ["ChainRulesCore", "LinearAlgebra"]
git-tree-sha1 = "69f7020bd72f069c219b5e8c236c1fa90d2cb409"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.2.1"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "8eaf9f1b4921132a4cff3f36a1d9ba923b14a481"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.4"

[[deps.AbstractTrees]]
git-tree-sha1 = "03e0550477d86222521d254b741d470ba17ea0b5"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.3.4"

[[deps.Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "195c5505521008abea5aee4f96930717958eac6f"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.4.0"

[[deps.Animations]]
deps = ["Colors"]
git-tree-sha1 = "e81c509d2c8e49592413bfb0bb3b08150056c79d"
uuid = "27a7e980-b3e6-11e9-2bcd-0b925532e340"
version = "0.4.1"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.ArrayInterface]]
deps = ["IfElse", "LinearAlgebra", "Requires", "SparseArrays", "Static"]
git-tree-sha1 = "d84c956c4c0548b4caf0e4e96cf5b6494b5b1529"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "3.1.32"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Automa]]
deps = ["Printf", "ScanByte", "TranscodingStreams"]
git-tree-sha1 = "d50976f217489ce799e366d9561d56a98a30d7fe"
uuid = "67c07d97-cdcb-5c2c-af73-a7f9c32a568b"
version = "0.8.2"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "66771c8d21c8ff5e3a93379480a2307ac36863f7"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.0.1"

[[deps.AxisArrays]]
deps = ["Dates", "IntervalSets", "IterTools", "RangeArrays"]
git-tree-sha1 = "1dd4d9f5beebac0c03446918741b1a03dc5e5788"
uuid = "39de3d68-74b9-583c-8d2d-e117c070f3a9"
version = "0.4.6"

[[deps.BFloat16s]]
deps = ["LinearAlgebra", "Printf", "Random", "Test"]
git-tree-sha1 = "a598ecb0d717092b5539dbbe890c98bac842b072"
uuid = "ab4f0b2a-ad5b-11e8-123f-65d77653426b"
version = "0.2.0"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.BenchmarkTools]]
deps = ["JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "d9a9701b899b30332bbcb3e1679c41cce81fb0e8"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.3.2"

[[deps.BufferedStreams]]
git-tree-sha1 = "bb065b14d7f941b8617bc323063dbe79f55d16ea"
uuid = "e1450e63-4bb3-523b-b2a4-4ffa8c0fd77d"
version = "1.1.0"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[deps.CEnum]]
git-tree-sha1 = "eb4cb44a499229b3b8426dcfb5dd85333951ff90"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.4.2"

[[deps.CUDA]]
deps = ["AbstractFFTs", "Adapt", "BFloat16s", "CEnum", "CompilerSupportLibraries_jll", "ExprTools", "GPUArrays", "GPUCompiler", "LLVM", "LazyArtifacts", "Libdl", "LinearAlgebra", "Logging", "Printf", "Random", "Random123", "RandomNumbers", "Reexport", "Requires", "SparseArrays", "SpecialFunctions", "TimerOutputs"]
git-tree-sha1 = "49549e2c28ffb9cc77b3689dc10e46e6271e9452"
uuid = "052768ef-5323-5732-b1bb-66c8b64840ba"
version = "3.12.0"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b859a208b2397a7a623a03449e4636bdb17bcf2"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+1"

[[deps.Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[deps.ChainRules]]
deps = ["Adapt", "ChainRulesCore", "Compat", "Distributed", "GPUArraysCore", "IrrationalConstants", "LinearAlgebra", "Random", "RealDot", "SparseArrays", "Statistics", "StructArrays"]
git-tree-sha1 = "0c8c8887763f42583e1206ee35413a43c91e2623"
uuid = "082447d4-558c-5d27-93f4-14fc19e9eca2"
version = "1.45.0"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "e7ff6cadf743c098e08fca25c91103ee4303c9bb"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.15.6"

[[deps.ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "38f7a08f19d8810338d4f5085211c7dfa5d5bdd8"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.4"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "ded953804d019afa9a3f98981d99b33e3db7b6da"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.0"

[[deps.ColorBrewer]]
deps = ["Colors", "JSON", "Test"]
git-tree-sha1 = "61c5334f33d91e570e1d0c3eb5465835242582c4"
uuid = "a2cac450-b92f-5266-8821-25eda20663c8"
version = "0.4.0"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "Random", "SnoopPrecompile"]
git-tree-sha1 = "aa3edc8f8dea6cbfa176ee12f7c2fc82f0608ed3"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.20.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "SpecialFunctions", "Statistics", "TensorCore"]
git-tree-sha1 = "d08c20eef1f2cbc6e60fd3612ac4340b89fea322"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.9.9"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[deps.CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[deps.Compat]]
deps = ["Dates", "LinearAlgebra", "UUIDs"]
git-tree-sha1 = "3ca828fe1b75fa84b021a7860bd039eaea84d2f2"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.3.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "0.5.2+0"

[[deps.Contour]]
git-tree-sha1 = "d05d9e7b7aedff4e5b51a029dced05cfb6125781"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.2"

[[deps.DataAPI]]
git-tree-sha1 = "e08915633fcb3ea83bf9d6126292e5bc5c739922"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.13.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "d1fff3a548102f48987a52a2e0d114fa97d730f0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.13"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[deps.DensityInterface]]
deps = ["InverseFunctions", "Test"]
git-tree-sha1 = "80c3e8639e3353e5d2912fb3a1916b8455e2494b"
uuid = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
version = "0.4.0"

[[deps.DiffResults]]
deps = ["StaticArraysCore"]
git-tree-sha1 = "782dd5f4561f5d267313f23853baaaa4c52ea621"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.1.0"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "c5b6685d53f933c11404a3ae9822afe30d522494"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.12.2"

[[deps.Distances]]
deps = ["LinearAlgebra", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "3258d0659f812acde79e8a74b11f17ac06d0ca04"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.7"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Distributions]]
deps = ["ChainRulesCore", "DensityInterface", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "Test"]
git-tree-sha1 = "bee795cdeabc7601776abbd6b9aac2ca62429966"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.77"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "c36550cb29cbe373e95b3f40486b9a4148f89ffd"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.2"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.DualNumbers]]
deps = ["Calculus", "NaNMath", "SpecialFunctions"]
git-tree-sha1 = "5837a837389fccf076445fce071c8ddaea35a566"
uuid = "fa6b7ba4-c1ee-5f82-b5fc-ecf0adba8f74"
version = "0.6.8"

[[deps.EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e3290f2d49e661fbd94046d7e3726ffcb2d41053"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.4+0"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bad72f730e9e91c08d9427d5e8db95478a3c323d"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.4.8+0"

[[deps.ExprTools]]
git-tree-sha1 = "56559bbef6ca5ea0c0818fa5c90320398a6fbf8d"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.8"

[[deps.Extents]]
git-tree-sha1 = "5e1e4c53fa39afe63a7d356e30452249365fba99"
uuid = "411431e0-e8b7-467b-b5e0-f676ba4f2910"
version = "0.1.1"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Pkg", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "74faea50c1d007c85837327f6775bea60b5492dd"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.2+2"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "90630efff0894f8142308e334473eba54c433549"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.5.0"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c6033cc3892d0ef5bb9cd29b7f2f0331ea5184ea"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+0"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "7be5f99f7d15578798f338f5433b6c432ea8037b"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.16.0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "802bfc139833d2ba893dd9e62ba1767c88d708ae"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.13.5"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Flux]]
deps = ["AbstractTrees", "Adapt", "ArrayInterface", "CUDA", "CodecZlib", "Colors", "DelimitedFiles", "Functors", "Juno", "LinearAlgebra", "MacroTools", "NNlib", "NNlibCUDA", "Pkg", "Printf", "Random", "Reexport", "SHA", "SparseArrays", "Statistics", "StatsBase", "Test", "ZipFile", "Zygote"]
git-tree-sha1 = "511b7c48eebb602a8f63e7d6c63e25633468dc16"
uuid = "587475ba-b771-5e3f-ad9e-33799f191a9c"
version = "0.12.10"

[[deps.Flux3D]]
deps = ["CUDA", "Distributions", "FileIO", "Flux", "GeometryBasics", "LinearAlgebra", "MeshIO", "Meshing", "NearestNeighbors", "Printf", "Requires", "SHA", "SparseArrays", "Statistics", "Zygote"]
git-tree-sha1 = "96a2cd6f4a07fe991e7cd24f66e5ed1c950cd8aa"
uuid = "432009dd-59a1-4b72-8c93-6462ce9b220f"
version = "0.1.6"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[deps.Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions", "StaticArrays"]
git-tree-sha1 = "10fa12fe96e4d76acfa738f4df2126589a67374f"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.33"

[[deps.FreeType]]
deps = ["CEnum", "FreeType2_jll"]
git-tree-sha1 = "cabd77ab6a6fdff49bfd24af2ebe76e6e018a2b4"
uuid = "b38be410-82b0-50bf-ab77-7b57e271db43"
version = "4.0.0"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "87eb71354d8ec1a96d4a7636bd57a7347dde3ef9"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.4+0"

[[deps.FreeTypeAbstraction]]
deps = ["ColorVectorSpace", "Colors", "FreeType", "GeometryBasics"]
git-tree-sha1 = "38a92e40157100e796690421e34a11c107205c86"
uuid = "663a7486-cb36-511b-a19d-713bb74d65c9"
version = "0.10.0"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[deps.Functors]]
git-tree-sha1 = "223fffa49ca0ff9ce4f875be001ffe173b2b7de4"
uuid = "d9f16b24-f501-4c13-a1f2-28368ffc5196"
version = "0.2.8"

[[deps.GLFW]]
deps = ["GLFW_jll"]
git-tree-sha1 = "35dbc482f0967d8dceaa7ce007d16f9064072166"
uuid = "f7f18e0c-5ee9-5ccd-a5bf-e8befd85ed98"
version = "3.4.1"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "d972031d28c8c8d9d7b41a536ad7bb0c2579caca"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.8+0"

[[deps.GLMakie]]
deps = ["ColorTypes", "Colors", "FileIO", "FixedPointNumbers", "FreeTypeAbstraction", "GLFW", "GeometryBasics", "LinearAlgebra", "Makie", "Markdown", "MeshIO", "ModernGL", "Observables", "Printf", "Serialization", "ShaderAbstractions", "SnoopPrecompile", "StaticArrays"]
git-tree-sha1 = "f4416a0c2aedf3e9e18ba1c9495f382ae7af74d1"
uuid = "e9467ef8-e4e7-5192-8a1a-b1aee30e663a"
version = "0.7.3"

[[deps.GPUArrays]]
deps = ["Adapt", "GPUArraysCore", "LLVM", "LinearAlgebra", "Printf", "Random", "Reexport", "Serialization", "Statistics"]
git-tree-sha1 = "45d7deaf05cbb44116ba785d147c518ab46352d7"
uuid = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"
version = "8.5.0"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "6872f5ec8fd1a38880f027a26739d42dcda6691f"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.1.2"

[[deps.GPUCompiler]]
deps = ["ExprTools", "InteractiveUtils", "LLVM", "Libdl", "Logging", "TimerOutputs", "UUIDs"]
git-tree-sha1 = "76f70a337a153c1632104af19d29023dbb6f30dd"
uuid = "61eb1bfa-7361-4325-ad38-22787b887f55"
version = "0.16.6"

[[deps.GeoInterface]]
deps = ["Extents"]
git-tree-sha1 = "fb28b5dc239d0174d7297310ef7b84a11804dfab"
uuid = "cf35fbd7-0cd7-5166-be24-54bfbe79505f"
version = "1.0.1"

[[deps.GeometryBasics]]
deps = ["EarCut_jll", "GeoInterface", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "fe9aea4ed3ec6afdfbeb5a4f39a2208909b162a6"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.5"

[[deps.GeometryTypes]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "StaticArrays"]
git-tree-sha1 = "d796f7be0383b5416cd403420ce0af083b0f9b28"
uuid = "4d00f742-c7ba-57c2-abde-4428a4b178cb"
version = "0.8.5"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "fb83fbe02fe57f2c068013aa94bcdf6760d3a7a7"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.74.0+1"

[[deps.Graphics]]
deps = ["Colors", "LinearAlgebra", "NaNMath"]
git-tree-sha1 = "d61890399bc535850c4bf08e4e0d3a7ad0f21cbd"
uuid = "a2bd30eb-e257-5431-a919-1863eab51364"
version = "1.1.2"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[deps.GridLayoutBase]]
deps = ["GeometryBasics", "InteractiveUtils", "Observables"]
git-tree-sha1 = "678d136003ed5bceaab05cf64519e3f956ffa4ba"
uuid = "3955a311-db13-416c-9275-1d80ed98e5e9"
version = "0.9.1"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HDF5]]
deps = ["Compat", "HDF5_jll", "Libdl", "Mmap", "Random", "Requires"]
git-tree-sha1 = "19effd6b5af759c8aaeb9c77f89422d3f975ab65"
uuid = "f67ccb44-e63f-5c2f-98bd-6dc0ccc4ba2f"
version = "0.16.12"

[[deps.HDF5_jll]]
deps = ["Artifacts", "JLLWrappers", "LibCURL_jll", "Libdl", "OpenSSL_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "4cc2bb72df6ff40b055295fdef6d92955f9dede8"
uuid = "0234f1f7-429e-5d53-9886-15a909be8d59"
version = "1.12.2+2"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[deps.HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "OpenLibm_jll", "SpecialFunctions", "Test"]
git-tree-sha1 = "709d864e3ed6e3545230601f94e11ebc65994641"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.11"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "c47c5fa4c5308f27ccaac35504858d8914e102f9"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.4"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

[[deps.IRTools]]
deps = ["InteractiveUtils", "MacroTools", "Test"]
git-tree-sha1 = "2e99184fca5eb6f075944b04c22edec29beb4778"
uuid = "7869d1d1-7146-5819-86e3-90919afe41df"
version = "0.4.7"

[[deps.IfElse]]
git-tree-sha1 = "debdd00ffef04665ccbb3e150747a77560e8fad1"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.1"

[[deps.ImageAxes]]
deps = ["AxisArrays", "ImageBase", "ImageCore", "Reexport", "SimpleTraits"]
git-tree-sha1 = "c54b581a83008dc7f292e205f4c409ab5caa0f04"
uuid = "2803e5a7-5153-5ecf-9a86-9b4c37f5f5ac"
version = "0.6.10"

[[deps.ImageBase]]
deps = ["ImageCore", "Reexport"]
git-tree-sha1 = "b51bb8cae22c66d0f6357e3bcb6363145ef20835"
uuid = "c817782e-172a-44cc-b673-b171935fbb9e"
version = "0.1.5"

[[deps.ImageCore]]
deps = ["AbstractFFTs", "ColorVectorSpace", "Colors", "FixedPointNumbers", "Graphics", "MappedArrays", "MosaicViews", "OffsetArrays", "PaddedViews", "Reexport"]
git-tree-sha1 = "acf614720ef026d38400b3817614c45882d75500"
uuid = "a09fc81d-aa75-5fe9-8630-4744c3626534"
version = "0.9.4"

[[deps.ImageIO]]
deps = ["FileIO", "IndirectArrays", "JpegTurbo", "LazyModules", "Netpbm", "OpenEXR", "PNGFiles", "QOI", "Sixel", "TiffImages", "UUIDs"]
git-tree-sha1 = "342f789fd041a55166764c351da1710db97ce0e0"
uuid = "82e4d734-157c-48bb-816b-45c225c6df19"
version = "0.6.6"

[[deps.ImageMetadata]]
deps = ["AxisArrays", "ImageAxes", "ImageBase", "ImageCore"]
git-tree-sha1 = "36cbaebed194b292590cba2593da27b34763804a"
uuid = "bc367c6b-8a6b-528e-b4bd-a4b897500b49"
version = "0.9.8"

[[deps.Imath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "87f7662e03a649cffa2e05bf19c303e168732d3e"
uuid = "905a6f67-0a94-5f89-b386-d35d92009cd1"
version = "3.1.2+0"

[[deps.IndirectArrays]]
git-tree-sha1 = "012e604e1c7458645cb8b436f8fba789a51b257f"
uuid = "9b13fd28-a010-5f03-acff-a1bbcff69959"
version = "1.0.0"

[[deps.Inflate]]
git-tree-sha1 = "5cd07aab533df5170988219191dfad0519391428"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.3"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d979e54b71da82f3a65b62553da4fc3d18c9004c"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2018.0.3+2"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.Interpolations]]
deps = ["Adapt", "AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "842dd89a6cb75e02e85fdd75c760cdc43f5d6863"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.14.6"

[[deps.IntervalSets]]
deps = ["Dates", "Random", "Statistics"]
git-tree-sha1 = "3f91cd3f56ea48d4d2a75c2a65455c5fc74fa347"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.7.3"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "49510dfcb407e572524ba94aeae2fced1f3feb0f"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.8"

[[deps.IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[deps.Isoband]]
deps = ["isoband_jll"]
git-tree-sha1 = "f9b6d97355599074dc867318950adaa6f9946137"
uuid = "f1662d9f-8043-43de-a69a-05efc1cc6ff4"
version = "0.1.1"

[[deps.IterTools]]
git-tree-sha1 = "fa6287a4469f5e048d763df38279ee729fbd44e5"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.4.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "3c837543ddb02250ef42f4738347454f95079d4e"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.3"

[[deps.JpegTurbo]]
deps = ["CEnum", "FileIO", "ImageCore", "JpegTurbo_jll", "TOML"]
git-tree-sha1 = "a77b273f1ddec645d1b7c4fd5fb98c8f90ad10a5"
uuid = "b835a17e-a41a-41e7-81f0-2f016b05efe0"
version = "0.1.1"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b53380851c6e6664204efb2e62cd24fa5c47e4ba"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.2+0"

[[deps.Juno]]
deps = ["Base64", "Logging", "Media", "Profile"]
git-tree-sha1 = "07cb43290a840908a771552911a6274bc6c072c7"
uuid = "e5e0dc1b-0480-54bc-9374-aad01c23163d"
version = "0.8.4"

[[deps.KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTW", "Interpolations", "StatsBase"]
git-tree-sha1 = "9816b296736292a80b9a3200eb7fbb57aaa3917a"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.5"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[deps.LLVM]]
deps = ["CEnum", "LLVMExtra_jll", "Libdl", "Printf", "Unicode"]
git-tree-sha1 = "e7e9184b0bf0158ac4e4aa9daf00041b5909bf1a"
uuid = "929cbde3-209d-540e-8aea-75f648917ca0"
version = "4.14.0"

[[deps.LLVMExtra_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg", "TOML"]
git-tree-sha1 = "771bfe376249626d3ca12bcd58ba243d3f961576"
uuid = "dad2f222-ce93-54a1-a47d-0025e8a3acab"
version = "0.0.16+0"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[deps.LazyModules]]
git-tree-sha1 = "a560dd966b386ac9ae60bdd3a3d3a326062d3c3e"
uuid = "8cdb02fc-e678-4876-92c5-9defec4f444e"
version = "0.3.1"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.3"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "7.84.0+0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.10.2+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "6f73d1dd803986947b2c750138528a999a6c7733"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.6.0+0"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "42b62845d70a619f063a7da093d995ec8e15e778"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+1"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "94d9c52ca447e23eac0c0f074effbcd38830deb5"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.18"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.MAT]]
deps = ["BufferedStreams", "CodecZlib", "HDF5", "SparseArrays"]
git-tree-sha1 = "971be550166fe3f604d28715302b58a3f7293160"
uuid = "23992714-dd62-5051-b70f-ba57cb901cac"
version = "0.10.3"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg"]
git-tree-sha1 = "2ce8695e1e699b68702c03402672a69f54b8aca9"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2022.2.0+0"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "42324d08725e200c23d4dfb549e0d5d89dede2d2"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.10"

[[deps.Makie]]
deps = ["Animations", "Base64", "ColorBrewer", "ColorSchemes", "ColorTypes", "Colors", "Contour", "Distributions", "DocStringExtensions", "FFMPEG", "FileIO", "FixedPointNumbers", "Formatting", "FreeType", "FreeTypeAbstraction", "GeometryBasics", "GridLayoutBase", "ImageIO", "InteractiveUtils", "IntervalSets", "Isoband", "KernelDensity", "LaTeXStrings", "LinearAlgebra", "MakieCore", "Markdown", "Match", "MathTeXEngine", "MiniQhull", "Observables", "OffsetArrays", "Packing", "PlotUtils", "PolygonOps", "Printf", "Random", "RelocatableFolders", "Serialization", "Showoff", "SignedDistanceFields", "SnoopPrecompile", "SparseArrays", "Statistics", "StatsBase", "StatsFuns", "StructArrays", "TriplotBase", "UnicodeFun"]
git-tree-sha1 = "d3b9553c2f5e0ca588e4395a9508cef024bd9e8a"
uuid = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
version = "0.18.3"

[[deps.MakieCore]]
deps = ["Observables"]
git-tree-sha1 = "c1885d865632e7f37e5a1489a164f44c54fb80c9"
uuid = "20f20a25-4f0e-4fdf-b5d1-57303727442b"
version = "0.5.2"

[[deps.MappedArrays]]
git-tree-sha1 = "e8b359ef06ec72e8c030463fe02efe5527ee5142"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.1"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.Match]]
git-tree-sha1 = "1d9bc5c1a6e7ee24effb93f175c9342f9154d97f"
uuid = "7eb4fadd-790c-5f42-8a69-bfa0b872bfbf"
version = "1.2.0"

[[deps.MathTeXEngine]]
deps = ["AbstractTrees", "Automa", "DataStructures", "FreeTypeAbstraction", "GeometryBasics", "LaTeXStrings", "REPL", "RelocatableFolders", "Test", "UnicodeFun"]
git-tree-sha1 = "f04120d9adf4f49be242db0b905bea0be32198d1"
uuid = "0a4f8689-d25c-4efe-a92b-7142dfc1aa53"
version = "0.5.4"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.0+0"

[[deps.Media]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "75a54abd10709c01f1b86b84ec225d26e840ed58"
uuid = "e89f7d12-3494-54d1-8411-f7d8b9ae1f27"
version = "0.5.0"

[[deps.MeshIO]]
deps = ["ColorTypes", "FileIO", "GeometryBasics", "Printf"]
git-tree-sha1 = "8be09d84a2d597c7c0c34d7d604c039c9763e48c"
uuid = "7269a6da-0436-5bbc-96c2-40638cbb6118"
version = "0.4.10"

[[deps.Meshing]]
deps = ["GeometryBasics", "GeometryTypes", "StaticArrays"]
git-tree-sha1 = "b32d34f3e3ca44391ca7261ca4eb96af71e022b9"
uuid = "e6723b4c-ebff-59f1-b4b7-d97aa5274f73"
version = "0.5.7"

[[deps.MiniQhull]]
deps = ["QhullMiniWrapper_jll"]
git-tree-sha1 = "9dc837d180ee49eeb7c8b77bb1c860452634b0d1"
uuid = "978d7f02-9e05-4691-894f-ae31a51d76ca"
version = "0.4.0"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.ModernGL]]
deps = ["Libdl"]
git-tree-sha1 = "cb7f10a4e69af6d022747ff181e3976f3de34408"
uuid = "66fc600b-dfda-50eb-8b99-91cfa97b1301"
version = "1.1.6"

[[deps.MosaicViews]]
deps = ["MappedArrays", "OffsetArrays", "PaddedViews", "StackViews"]
git-tree-sha1 = "7b86a5d4d70a9f5cdf2dacb3cbe6d251d1a61dbe"
uuid = "e94cdb99-869f-56ef-bcf0-1ae2bcbe0389"
version = "0.3.4"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.2.1"

[[deps.NNlib]]
deps = ["Adapt", "ChainRulesCore", "LinearAlgebra", "Pkg", "Requires", "Statistics"]
git-tree-sha1 = "00bcfcea7b2063807fdcab2e0ce86ef00b8b8000"
uuid = "872c559c-99b0-510c-b3b7-b6c96a88d5cd"
version = "0.8.10"

[[deps.NNlibCUDA]]
deps = ["Adapt", "CUDA", "LinearAlgebra", "NNlib", "Random", "Statistics"]
git-tree-sha1 = "4429261364c5ea5b7308aecaa10e803ace101631"
uuid = "a00861dc-f156-4864-bf3c-e6376f28a68d"
version = "0.2.4"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "a7c3d1da1189a1c2fe843a3bfa04d18d20eb3211"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.1"

[[deps.NearestNeighbors]]
deps = ["Distances", "StaticArrays"]
git-tree-sha1 = "440165bf08bc500b8fe4a7be2dc83271a00c0716"
uuid = "b8a86587-4115-5ab1-83bc-aa920d37bbce"
version = "0.4.12"

[[deps.Netpbm]]
deps = ["FileIO", "ImageCore", "ImageMetadata"]
git-tree-sha1 = "5ae7ca23e13855b3aba94550f26146c01d259267"
uuid = "f09324ee-3d7c-5217-9330-fc30815ba969"
version = "1.1.0"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.Observables]]
git-tree-sha1 = "6862738f9796b3edc1c09d0890afce4eca9e7e93"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.5.4"

[[deps.OffsetArrays]]
deps = ["Adapt"]
git-tree-sha1 = "f71d8950b724e9ff6110fc948dff5a329f901d64"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.12.8"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.20+0"

[[deps.OpenEXR]]
deps = ["Colors", "FileIO", "OpenEXR_jll"]
git-tree-sha1 = "327f53360fdb54df7ecd01e96ef1983536d1e633"
uuid = "52e1d378-f018-4a11-a4be-720524705ac7"
version = "0.3.2"

[[deps.OpenEXR_jll]]
deps = ["Artifacts", "Imath_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "923319661e9a22712f24596ce81c54fc0366f304"
uuid = "18a262bb-aa17-5467-a713-aee519bc75cb"
version = "3.1.1+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+0"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6e9dba33f9f2c44e08a020b0caf6903be540004"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.19+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.40.0+0"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "cf494dca75a69712a72b80bc48f59dcf3dea63ec"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.16"

[[deps.PNGFiles]]
deps = ["Base64", "CEnum", "ImageCore", "IndirectArrays", "OffsetArrays", "libpng_jll"]
git-tree-sha1 = "f809158b27eba0c18c269cf2a2be6ed751d3e81d"
uuid = "f57f5aa1-a3ce-4bc8-8ab9-96f992907883"
version = "0.3.17"

[[deps.Packing]]
deps = ["GeometryBasics"]
git-tree-sha1 = "1155f6f937fa2b94104162f01fa400e192e4272f"
uuid = "19eb6ba3-879d-56ad-ad62-d5c202156566"
version = "0.4.2"

[[deps.PaddedViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "03a7a85b76381a3d04c7a1656039197e70eda03d"
uuid = "5432bcbf-9aad-5242-b902-cca2824c8663"
version = "0.5.11"

[[deps.Parsers]]
deps = ["Dates", "SnoopPrecompile"]
git-tree-sha1 = "cceb0257b662528ecdf0b4b4302eb00e767b38e7"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.5.0"

[[deps.Pixman_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b4f5d02549a10e20780a24fce72bea96b6329e29"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.40.1+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.8.0"

[[deps.PkgVersion]]
deps = ["Pkg"]
git-tree-sha1 = "f6cf8e7944e50901594838951729a1861e668cb8"
uuid = "eebad327-c553-4316-9ea0-9fa01ccd7688"
version = "0.3.2"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "Printf", "Random", "Reexport", "SnoopPrecompile", "Statistics"]
git-tree-sha1 = "21303256d239f6b484977314674aef4bb1fe4420"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.3.1"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "efc140104e6d0ae3e7e30d56c98c4a927154d684"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.48"

[[deps.PolygonOps]]
git-tree-sha1 = "77b3d3605fc1cd0b42d95eba87dfcd2bf67d5ff6"
uuid = "647866c9-e3ac-4575-94e7-e3d426903924"
version = "0.1.2"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "47e5f437cc0e7ef2ce8406ce1e7e24d44915f88d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.3.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Profile]]
deps = ["Printf"]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"

[[deps.ProfileCanvas]]
deps = ["Base64", "JSON", "Pkg", "Profile", "REPL"]
git-tree-sha1 = "e42571ce9a614c2fbebcaa8aab23bbf8865c624e"
uuid = "efd6af41-a80b-495e-886c-e51b0c7d77a3"
version = "0.1.6"

[[deps.ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "d7a7aef8f8f2d537104f170139553b14dfe39fe9"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.7.2"

[[deps.QOI]]
deps = ["ColorTypes", "FileIO", "FixedPointNumbers"]
git-tree-sha1 = "18e8f4d1426e965c7b532ddd260599e1510d26ce"
uuid = "4b34888f-f399-49d4-9bb3-47ed5cae4e65"
version = "1.0.0"

[[deps.QhullMiniWrapper_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Qhull_jll"]
git-tree-sha1 = "607cf73c03f8a9f83b36db0b86a3a9c14179621f"
uuid = "460c41e3-6112-5d7f-b78c-b6823adb3f2d"
version = "1.0.0+1"

[[deps.Qhull_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "238dd7e2cc577281976b9681702174850f8d4cbc"
uuid = "784f63db-0788-585a-bace-daefebcd302b"
version = "8.0.1001+0"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "97aa253e65b784fd13e83774cadc95b38011d734"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.6.0"

[[deps.Quaternions]]
deps = ["LinearAlgebra", "Random"]
git-tree-sha1 = "fcebf40de9a04c58da5073ec09c1c1e95944c79b"
uuid = "94ee1d12-ae83-5a48-8b1c-48b8ff168ae0"
version = "0.6.1"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Random123]]
deps = ["Random", "RandomNumbers"]
git-tree-sha1 = "7a1a306b72cfa60634f03a911405f4e64d1b718b"
uuid = "74087812-796a-5b5d-8853-05524746bad3"
version = "1.6.0"

[[deps.RandomNumbers]]
deps = ["Random", "Requires"]
git-tree-sha1 = "043da614cc7e95c703498a491e2c21f58a2b8111"
uuid = "e6cf234a-135c-5ec9-84dd-332b85af5143"
version = "1.5.3"

[[deps.RangeArrays]]
git-tree-sha1 = "b9039e93773ddcfc828f12aadf7115b4b4d225f5"
uuid = "b3c3ace0-ae52-54e7-9d0b-2c1406fd6b9d"
version = "0.3.2"

[[deps.Ratios]]
deps = ["Requires"]
git-tree-sha1 = "dc84268fe0e3335a62e315a3a7cf2afa7178a734"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.3"

[[deps.RealDot]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9f0a1b71baaf7650f4fa8a1d168c7fb6ee41f0c9"
uuid = "c1ae055f-0cd5-4b69-90a6-9a35b1a98df9"
version = "0.1.0"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "90bc7a7c96410424509e4263e277e43250c05691"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.0"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "bf3188feca147ce108c76ad82c2792c57abe7b1f"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.0"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "68db32dff12bb6127bac73c209881191bf0efbb7"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.3.0+0"

[[deps.Rotations]]
deps = ["LinearAlgebra", "Quaternions", "Random", "StaticArrays", "Statistics"]
git-tree-sha1 = "793b6ef92f9e96167ddbbd2d9685009e200eb84f"
uuid = "6038ab10-8711-5258-84ad-4b1120ba62dc"
version = "1.3.3"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SIMD]]
git-tree-sha1 = "bc12e315740f3a36a6db85fa2c0212a848bd239e"
uuid = "fdea26ae-647d-5447-a871-4b548cad5224"
version = "3.4.2"

[[deps.ScanByte]]
deps = ["Libdl", "SIMD"]
git-tree-sha1 = "2436b15f376005e8790e318329560dcc67188e84"
uuid = "7b38b023-a4d7-4c5e-8d43-3f3097f304eb"
version = "0.3.3"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "f94f779c94e58bf9ea243e77a37e16d9de9126bd"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.1.1"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.ShaderAbstractions]]
deps = ["ColorTypes", "FixedPointNumbers", "GeometryBasics", "LinearAlgebra", "Observables", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "6b5bba824b515ec026064d1e7f5d61432e954b71"
uuid = "65257c39-d410-5151-9873-9b3e5be5013e"
version = "0.2.9"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SignedDistanceFields]]
deps = ["Random", "Statistics", "Test"]
git-tree-sha1 = "d263a08ec505853a5ff1c1ebde2070419e3f28e9"
uuid = "73760f76-fbc4-59ce-8f25-708e95d2df96"
version = "0.4.0"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "5d7e3f4e11935503d3ecaf7186eac40602e7d231"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.4"

[[deps.Sixel]]
deps = ["Dates", "FileIO", "ImageCore", "IndirectArrays", "OffsetArrays", "REPL", "libsixel_jll"]
git-tree-sha1 = "8fb59825be681d451c246a795117f317ecbcaa28"
uuid = "45858cf5-a6b0-47a3-bbea-62219f50df47"
version = "0.1.2"

[[deps.SnoopPrecompile]]
git-tree-sha1 = "f604441450a3c0569830946e5b33b78c928e1a85"
uuid = "66db9d55-30c0-4569-8b51-7e840670fc0c"
version = "1.0.1"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "d75bda01f8c31ebb72df80a46c88b25d1c79c56d"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.1.7"

[[deps.StackViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "46e589465204cd0c08b4bd97385e4fa79a0c770c"
uuid = "cae243ae-269e-4f55-b966-ac2d0dc13c15"
version = "0.1.1"

[[deps.Static]]
deps = ["IfElse"]
git-tree-sha1 = "a8f30abc7c64a39d389680b74e749cf33f872a70"
uuid = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"
version = "0.3.3"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "StaticArraysCore", "Statistics"]
git-tree-sha1 = "f86b3a049e5d05227b10e15dbb315c5b90f14988"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.5.9"

[[deps.StaticArraysCore]]
git-tree-sha1 = "6b7ba252635a5eff6a0b0664a41ee140a1c9e72a"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.0"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f9af7f195fb13589dd2e2d57fdb401717d2eb1f6"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.5.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "d1bf48bfcc554a3761a133fe3a9bb01488e06916"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.21"

[[deps.StatsFuns]]
deps = ["ChainRulesCore", "HypergeometricFunctions", "InverseFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "5783b877201a82fc0014cbf381e7e6eb130473a4"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.0.1"

[[deps.StructArrays]]
deps = ["Adapt", "DataAPI", "StaticArraysCore", "Tables"]
git-tree-sha1 = "13237798b407150a6d2e2bce5d793d7d9576e99e"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.13"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.0"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits", "Test"]
git-tree-sha1 = "c79322d36826aa2f4fd8ecfa96ddb47b174ac78d"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.10.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.1"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.TiffImages]]
deps = ["ColorTypes", "DataStructures", "DocStringExtensions", "FileIO", "FixedPointNumbers", "IndirectArrays", "Inflate", "Mmap", "OffsetArrays", "PkgVersion", "ProgressMeter", "UUIDs"]
git-tree-sha1 = "f8cd5b95aae14d3d88da725414bdde342457366f"
uuid = "731e570b-9d59-4bfa-96dc-6df516fadf69"
version = "0.6.2"

[[deps.TimerOutputs]]
deps = ["ExprTools", "Printf"]
git-tree-sha1 = "f2fd3f288dfc6f507b0c3a2eb3bac009251e548b"
uuid = "a759f4b9-e2f1-59dc-863e-4aeb61b1ea8f"
version = "0.5.22"

[[deps.TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "8a75929dcd3c38611db2f8d08546decb514fcadf"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.9"

[[deps.Tricks]]
git-tree-sha1 = "6bac775f2d42a611cdfcd1fb217ee719630c4175"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.6"

[[deps.TriplotBase]]
git-tree-sha1 = "4d4ed7f294cda19382ff7de4c137d24d16adc89b"
uuid = "981d1d27-644d-49a2-9326-4793e63143c3"
version = "0.1.0"

[[deps.URIs]]
git-tree-sha1 = "ac00576f90d8a259f2c9d823e91d1de3fd44d348"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.4.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "de67fa59e33ad156a590055375a30b23c40299d3"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "0.5.5"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "58443b63fb7e465a8a7210828c91c08b92132dff"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.9.14+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "5be649d550f3f4b95308bf0183b82e2582876527"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.6.9+4"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4e490d5c960c314f33885790ed410ff3a94ce67e"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.9+4"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fe47bd2247248125c428978740e18a681372dd4"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.3+4"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6783737e45d3c59a4a4c4091f5f88cdcf0908cbb"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.0+3"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "daf17f441228e7a3833846cd048892861cff16d6"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.13.0+3"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "79c31e7844f6ecf779705fbc12146eb190b7d845"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.4.0+3"

[[deps.ZipFile]]
deps = ["Libdl", "Printf", "Zlib_jll"]
git-tree-sha1 = "3593e69e469d2111389a9bd06bac1f3d730ac6de"
uuid = "a5390f91-8eb1-5f08-bee0-b1d1ffed6cea"
version = "0.9.4"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.12+3"

[[deps.Zygote]]
deps = ["AbstractFFTs", "ChainRules", "ChainRulesCore", "DiffRules", "Distributed", "FillArrays", "ForwardDiff", "GPUArrays", "GPUArraysCore", "IRTools", "InteractiveUtils", "LinearAlgebra", "LogExpFunctions", "MacroTools", "NaNMath", "Random", "Requires", "SparseArrays", "SpecialFunctions", "Statistics", "ZygoteRules"]
git-tree-sha1 = "66cc604b9a27a660e25a54e408b4371123a186a6"
uuid = "e88e6eb3-aa80-5325-afca-941959d7151f"
version = "0.6.49"

[[deps.ZygoteRules]]
deps = ["MacroTools"]
git-tree-sha1 = "8c1a8e4dfacb1fd631745552c8db35d0deb09ea0"
uuid = "700de1a5-db45-46bc-99cf-38207098b444"
version = "0.2.2"

[[deps.isoband_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51b5eeb3f98367157a7a12a1fb0aa5328946c03c"
uuid = "9a68df92-36a6-505f-a73e-abb412b6bfb4"
version = "0.2.3+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3a2ea60308f0996d26f1e5354e10c24e9ef905d4"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.4.0+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.1.1+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[deps.libsixel_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Pkg", "libpng_jll"]
git-tree-sha1 = "d4f63314c8aa1e48cd22aa0c17ed76cd1ae48c3c"
uuid = "075b6546-f08a-558a-be8f-8157d0f608a5"
version = "1.10.3+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "b910cb81ef3fe6e78bf6acee440bda86fd6ae00c"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+1"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.48.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"
"""

# ╔═╡ Cell order:
# ╟─5436358a-86ed-4724-a4b1-58ed3cb18d32
# ╠═e4532490-63e8-11ed-2f97-632853f48f1d
# ╠═5261d83a-b73f-4a7f-a101-5c57b47d29d1
# ╟─012ef84b-48cb-4932-9879-48acc2e1f39d
# ╠═ab5a6a07-d2b5-43fa-9ca2-98ea2c9438ad
# ╠═8f05aef0-903e-4c20-8fad-3ee03e43dda2
# ╟─fa55bd44-d803-41f8-a48e-f7bc03b66dec
# ╠═b0ef0120-4385-457f-8104-217de22ba4fa
# ╟─191c1a65-d627-4595-88df-d5b5c73edcdf
# ╠═654f8ec6-ee3a-4570-b122-03cab1955c47
# ╟─c462804d-5ea6-4fb7-baf9-861c9c961fe7
# ╠═2350385e-81e0-47ce-b13d-0cbbbfed4894
# ╠═aba902a4-7dd2-42cc-84bc-62c6c5957e4b
# ╠═c7fdedf4-0ece-4cc8-8752-fbfd78696d09
# ╠═6b7645ec-03e2-4eec-83bd-bac1650c5c1a
# ╠═3c852557-bdfe-4c64-85b5-c8ef84333238
# ╠═3c520c56-d555-4749-92ff-28ea3b541160
# ╟─11aebf61-cf36-450f-aa36-af3508844553
# ╠═521daff7-e4dc-43b3-aa7c-3e543c5b6ffe
# ╟─f39ff907-bc7c-49bd-b813-65ad03f4b190
# ╠═bf054703-1880-4b01-8cd8-35fcf7c37973
# ╠═1eee9776-8495-45dc-86dc-b05c16bea058
# ╠═77986ac9-66ed-46b1-9a2f-e9a7dfa812d2
# ╠═5674df53-78ab-404e-859b-7b8abdaad2b3
# ╠═ec83c06c-7480-4b27-8f42-b0794330657a
# ╠═0388e7b0-95d7-46b9-9a37-00180052d6dc
# ╠═ab7d59c1-e7fa-45f9-bb84-0af2bf8b8fce
# ╠═bc29f7cd-5f96-491f-ac03-dd0f01f574ae
# ╠═1e902be5-c98d-421a-8ef4-7294e6855640
# ╠═8a7ce52a-403f-45e7-b9b8-b4b5b46c69ac
# ╟─56844bba-6900-4c19-8925-03d5ae307599
# ╠═01a9c86d-f0b2-4a6c-ae07-0f7c0090ca91
# ╠═bfd6c10c-04da-4e52-a0f2-1641e4572317
# ╠═fbdbf4fd-bd9e-4844-890a-a7731279089d
# ╟─65e95d65-c0ec-4568-a52e-3b9242f68494
# ╟─d1886891-e260-4104-866a-ead8284af0ce
# ╟─53f8e4b5-d39b-44a6-b8d7-017a94883293
# ╟─22dd669d-f9ec-4b54-b161-7a4ffa8ef708
# ╟─b7bb98d7-9469-4611-ab27-ddca18b9cfb5
# ╟─26fece0f-2e3c-4966-81fa-ce794b2079c6
# ╟─33020dfe-eaef-47b6-800f-8329109de36b
# ╟─433792e0-0877-4ac8-971a-978e4fcf60bd
# ╠═16e90931-721e-4dbf-b921-b68f119a4476
# ╠═7b50b05c-b3bf-468c-95da-7ddcf72479a0
# ╠═9eba2bdd-1ccf-4621-8df0-9ed3eec8707a
# ╟─9f17db92-7724-4459-bc50-1a8fa27944ac
# ╠═832d9d3d-5379-49a9-8a82-6486a835573e
# ╠═0f8abea4-3fdc-4b80-9a98-c4b054433b18
# ╟─1d1a3f08-ede0-4cea-8aea-a5aee6aaeaed
# ╠═fa1fd339-2afe-467c-90f7-02bc0410c7d7
# ╠═737d6393-a015-4037-bc91-ff93e25fcdf5
# ╠═bfac69ec-9f99-46f8-b379-15ff98f8799d
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
