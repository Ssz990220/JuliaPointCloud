using Random, LinearAlgebra, BenchmarkTools, Test, Statistics, StaticArrays
using NearestNeighbors
using StaticArrays
using Rotations
using ProfileCanvas
using MeshIO,FileIO,GeometryBasics

@inline SRange(n) = StaticArrays.SUnitRange(n,n)

"""
    load_PC(Path)
Load a point cloud from given Path. Returns a point cloud `P` and # of Points `n`
"""
function load_PC(Path)
	PC = load(Path).position
    n = size(PC,1)
	D = size(first(PC),1);
	T = eltype(first(PC));
	P = SMatrix{D,1,T,D}.(PC);	## Format Conversion
    return P, n
end

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

function Point2PC(P::Vector{GeometryBasics.Point{D,T}}) where {D,T}
	n = size(P,1)
	PC = zeros(T,D,n)
	for i = 1:n
		PC[:,i] = P[i]
	end
	return PC
end

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

"""
	get_pc_closest_point!(X,Y,Tree,Q,W)
Find the index & point coordiante of the closest point for each point in `X::Vector{SMatrix}` from KDTree `Tree::KDTree`, which is built from point cloud `Y`. The indices are filled in `W`. Points are assigned in `Q`
"""
function get_pc_closest_point!(X::Vector{SMatrix{D,1,T,D}},Y::Vector{SMatrix{D,1,T,D}},Tree,Q::Vector{SMatrix{D,1,T,D}},W::Vector{T}) where{T,D}
	@inbounds Threads.@threads for i ??? eachindex(X)
		idx,dist = nn(Tree,X[i])
		W[i] = dist[1]
		Q[i] = Y[idx[1]]
	end
	return Q,W
end


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

begin
"""
	transform_PC(PC,R,t)
Transforming a Point Cloud `PC` under rotation `R` and translation `t`

See also [`transform_PC!`](@ref)
"""
	function transform_PC(PC::Vector{SMatrix{3,1,T,3}}, R::T???, t::T???) where {T<:Number,T???<:AbstractMatrix{T},T???<:AbstractMatrix{T}}
		n = size(PC,1)
		PC_ = Array{SMatrix{3,1,T,3}}(undef,n)
		Threads.@threads for i ??? eachindex(PC)
			@inbounds PC_[i] = R*PC[i] .+ t;
		end
		return PC_
	end
	function transform_PC(PC::Vector{SMatrix{3,1,T???,3}}, T::T???) where {T???<:Number,T???<:AbstractMatrix{T???}}
		n = size(PC,1)
		R,t = T2rt(T)
		PC_ = Array{SMatrix{3,1,T???,3}}(undef,n)
		Threads.@threads for i ??? eachindex(PC)
			@inbounds PC_[i] = R*PC[i] .+ t;
		end
		return PC_
	end
end

"""
	transform_PC!(PC,R,t)
Inplace transformation of a Point Cloud `PC` under rotation `R` and translation `t`

See also [`transform_PC`](@ref)
"""
function transform_PC!(PC::Vector{SMatrix{3,1,T,3}}, R::T???, t::T???) where {T<:Number,T???<:AbstractMatrix{T},T???<:AbstractMatrix{T}}
	Threads.@threads for i ??? eachindex(PC)
		@inbounds PC[i] = R*PC[i] .+ t;
	end
end

"""
	transform_PC!(PC,T)
Inplace transformation of a Point Cloud `PC` under transformation `T`

See also [`transform_PC`](@ref)
"""
function transform_PC!(PC::Vector{SMatrix{3,1,T???,3}}, T::T???) where {T???<:Number,T???<:AbstractMatrix{T???}}
	R,t = T2rt(T)
	Threads.@threads for i ??? eachindex(PC)
		@inbounds PC[i] = R*PC[i] .+ t;
	end
end

"""
	P2P_ICP(source,target, w)
Point to Point ICP with SVD. **Points in source and points in target are listed correspoindingly, given weight `w`**
"""
function P2P_ICP(X::Vector{SMatrix{D,1,T,D}},Y::Vector{SMatrix{D,1,T,D}},w::Vector{T}=ones(T,size(X,1))) where {T<:Number,D}
	w??? = w/sum(w);
	mean??? = sum(X.*w???);X = X.-[mean???];
	mean??? = sum(Y.*w???);Y = Y.-[mean???];
	?? = reduce(hcat, X) * reduce(hcat, Y.*w???)'
	F = svd(??);
	U = SMatrix{D,D,T,D*D}(F.U);
	V = SMatrix{D,D,T,D*D}(F.V);
	if det(U)*det(V) < 0
		s = ones(T,D); s[end] = -s[end];
		S = SMatrix{D,D,T,D*D}(diagm(s))
		R = V*S*U';
	else
		R = V*U';
	end
	t = mean???-R*mean???
	X = X.+ [mean???]; Y = Y.+ [mean???];
	return rt2T(R,t)
end

"""
	rt2T(R,t)
Combine a rotation matrix `R` and a translation vector `t` to a homogenous matrix `T`
"""
function rt2T(R::T???,t::SMatrix{D,1,T,D}) where {D,T???<:AbstractMatrix,T<:Number}
	row = zeros(1,D+1); row[1,D+1] = T(1.0);
	row = SMatrix{1,D+1,T,D+1}(row);
	return vcat(hcat(R,t),row)
end

"""
	T2rt(T)
Decompose a homogenous transformation matrix `T` into a rotation matrix `R` and a translational vector `t`
"""
function T2rt(T)
	R = T[SOneTo(3),SOneTo(3)]
	t = T[SOneTo(3),4]
	return R,t
end

"""
	get_energy(W,?????,f)
get point cloud error energy given error `W`, parameter `?????`, with function specified by `f`
"""
function get_energy(W,?????,f)
	if f == "tukey"
		return tukey_energy(W,?????)
	elseif f == "fair"
		return fair_energy(W,?????)
	elseif f == "log"
		return logistic_energy(W,?????)
	elseif f == "trimmed"
		return trimmed_energy(W,?????)
	elseif f == "welsch"
		return welsch_energy(W,?????)
	elseif f == "auto_welsch"
		return autowelsch_energy(W,?????)
	elseif f == "uniform"
		return uniform_energy(W)
	else
		return uniform_energy(W)
	end
end

"""
	robust_weight!(W,?????,f)
get point cloud robust weight given error `W`, parameter `?????`, with function specified by `f`
"""
function robust_weight!(W,?????,f)
	if f == "tukey"
		W.= tukey_weight(W,?????)
	elseif f == "fair"
		W.=  fair_weight(W,?????)
	elseif f == "log"
		W.=  logistic_weight(W,?????)
	elseif f == "trimmed"
		W.=  trimmed_weight(W,?????)
	elseif f == "welsch"
		W.=  welsch_weight(W,?????)
	elseif f == "auto_welsch"
		W.=  autowelsch_weight(W,?????)
	elseif f == "uniform"
		W.=  uniform_weight(W)
	else
		W.=  uniform_weight(W)
	end
end

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
end

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
end

function FindKnearestMed(P::Vector{SMatrix{3,1,T,3}},Tree,k) where {T}
	n = size(P,1)
	Xnearest = Vector{T}(undef,n)
	Threads.@threads for i = 1:n
		idxs, dists = knn(Tree, P[i], k, true)
		Xnearest[i] = median(dists[1])
	end
	return sqrt(median(Xnearest))
end

"""
	median(v)
Find the median value of a vector `v`. `v` should be sorted ahead.
"""
function median(v::Vector{T}) where {T}
	n = size(v,1)
	if iseven(n)
		return (v[n??2] + v[n??2+1])/T(2.0) 
	else
		return (v[(n+1)??2] + v[(n-1)??2])/T(2.0)
	end
end

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

function wrapto1(a::T) where {T<:Number}
	if a > 1 
		return T(1.0)
	elseif a < -1
		return T(-1.0)
	else
		return a
	end
end

se32vec(se3) = @SMatrix [-se3[2,3];se3[1,3];-se3[1,2];se3[1,4];se3[2,4];se3[3,4]]

vec2se3(vec::StaticArray{Tuple{D, 1}, T, 2}) where {T,D}  = @SMatrix [T(0.0) -vec[3] vec[2] vec[4]; 
										vec[3] T(0.0) -vec[1] vec[5];
										-vec[2] vec[1] T(0.0) vec[6]; 
										T(0.0) T(0.0) T(0.0) T(0.0)]
