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

function FICP_P2P(source::Vector{SMatrix{D,1,T,D}},target::Vector{SMatrix{D,1,T,D}},par) where {T<:Number,D}
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

	## AA Buffers
	d = D == 3 ? 6 : 4 		# 3D & 2D only
	theta = @MMatrix zeros(T,1,par.aa.m)
	scales = @MMatrix zeros(T,1,par.aa.m)
	local U = @MMatrix zeros(T,d,1)
	local Tᵥ = @MMatrix zeros(T,d,1)
	local Gs = @MMatrix zeros(T,d,par.aa.m,)
	local Fs = @MMatrix zeros(T,d,par.aa.m,)
	local iter = 0
	local counter = 0

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
	end
	return Tₘ
end

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