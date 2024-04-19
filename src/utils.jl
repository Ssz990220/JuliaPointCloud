@inline SRange(n) = StaticArrays.SUnitRange(n, n)

"""
    load_PC(Path)
Load a point cloud from given Path. Returns a point cloud `P` and # of Points `n`
"""
function load_PC(Path)
    PC = load(Path).position
    n = size(PC, 1)
    D = size(first(PC), 1)
    T = eltype(first(PC))
    P = SVector{D,T}.(PC)## Format Conversion
    return P, n
end

"""
	downSample_PC(PC, n)
Downsample a point cloud `PC` to `n` points with random sampling
"""
function downSample_PC(PC::Vector{SVector{D,T}}, n::Int) where {T,D}
    nₚ = size(PC, 1)
    idx = randperm(nₚ)[1:n]
    return PC[idx]
end

"""
    PC2SVector(PC)
Convert a point cloud to array of StaticMatrices
"""
function PC2SVector(PC::Matrix{T}) where {T<:Float}
    n = size(PC, 2)
    PCVec = Vector{SVector{3,T}}(undef, n)
    Threads.@threads for i = 1:n
        @inbounds PCVec[i] = @SVector [PC[1, i] for i = 1:3]
    end
    return PCVec
end

function Point2PC(P::Vector{GeometryBasics.Point{D,T}}) where {D,T}
    n = size(P, 1)
    PC = zeros(T, D, n)
    for i = 1:n
        PC[:, i] = P[i]
    end
    return PC
end

"""
	Svector2PC(PC)
Convert vector of Points in SMatrix to a Matrix
"""
function SVector2PC(P::Vector{SVector{D,T}}) where {T<:Float,D}
    n = size(P, 1)
    PC = zeros(T, D, n)
    Threads.@threads for i = 1:n
        @inbounds PC[:, i] = P[i][:]'
    end
    return PC
end

"""
	get_pc_closest_point(X,Y,Tree)
Find the index & point coordiante of the closest point for each point in `X::Vector{SMatrix}` from KDTree `Tree::KDTree`, which is built from point cloud `Y`. The indices are filled in `W`. Points are assigned in `Q`
"""
function get_pc_closest_point(X::Vector{SVector{D,T}}, Y::Vector{SVector{D,T}}, Tree) where {T,D}
    n = size(X, 1)
    Q = Vector{SVector{D,T}}(undef, n)
    W = zeros(T, n)
    Threads.@threads for i = 1:n
        @inbounds idx, dist = nn(Tree, X[i])
        @inbounds W[i] = dist[1]
        @inbounds Q[i] = Y[idx[1]]
    end
    return Q, W
end

"""
	get_pc_closest_point!(X,Y,Tree,Q,W)
Find the index & point coordiante of the closest point for each point in `X::Vector{SMatrix}` from KDTree `Tree::KDTree`, which is built from point cloud `Y`. The indices are filled in `W`. Points are assigned in `Q`
"""
function get_pc_closest_point!(X::Vector{SVector{D,T}}, Y::Vector{SVector{D,T}}, Tree, Q::Vector{SVector{D,T}}, W::Vector{T}) where {T,D}
    @inbounds Threads.@threads for i ∈ eachindex(X)
        idx, dist = nn(Tree, X[i])
        W[i] = dist[1]
        Q[i] = Y[idx[1]]
    end
    return Q, W
end


"""
	get_pc_closest_point_id(PC,Tree,W)
Find the index of the closest point for each point in `P::Vector{SMatrix}` from KDTree `Tree::KDTree`. The indices are filled in `W`
"""
function get_pc_closest_point_id(P::Vector{SVector{D,T}}, Tree) where {T,D}
    n = size(P, 1)
    id = zeros(n)
    Threads.@threads for i = 1:n
        @inbounds idx = nn(Tree, P[i])[1][1]
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
    function transform_PC(PC::Vector{SVector{3,T}}, R::T₂, t::T₃) where {T<:Float,T₂<:SMatrix{3,3,T,9},T₃<:SVector{3,T}}
        n = size(PC, 1)
        PC_ = Vector{SVector{3,T}}(undef, n)
        Threads.@threads for i ∈ eachindex(PC)
            @inbounds PC_[i] = R * PC[i] .+ t
        end
        return PC_
    end

    function transform_PC(PC::Vector{SVector{2,T}}, R::T₂, t::T₃) where {T<:Float,T₂<:SMatrix{2,2,T,4},T₃<:SVector{2,T}}
        n = size(PC, 1)
        PC_ = Vector{SVector{2,T}}(undef, n)
        Threads.@threads for i ∈ eachindex(PC)
            @inbounds PC_[i] = R * PC[i] .+ t
        end
        return PC_
    end

    function transform_PC(PC::Vector{SVector{3,T₁}}, T::T₂) where {T₁<:Float,T₂<:SMatrix{4,4,T₁,16}}
        n = size(PC, 1)
        R, t = T2rt(T)
        PC_ = Vector{SVector{3,T₁}}(undef, n)
        Threads.@threads for i ∈ eachindex(PC)
            @inbounds PC_[i] = R * PC[i] .+ t
        end
        return PC_
    end

    function transform_PC(PC::Vector{SVector{2,T₁}}, T::T₂) where {T₁<:Float,T₂<:SMatrix{3,3,T₁,9}}
        n = size(PC, 1)
        R, t = T2rt(T)
        PC_ = Vector{SVector{2,T₁}}(undef, n)
        Threads.@threads for i ∈ eachindex(PC)
            @inbounds PC_[i] = R * PC[i] .+ t
        end
        return PC_
    end
end

"""
	transform_PC!(PC,R,t)
Inplace transformation of a Point Cloud `PC` under rotation `R` and translation `t`

See also [`transform_PC`](@ref)
"""
function transform_PC!(PC::Vector{SVector{3,T}}, R::T₂, t::T₃) where {T<:Float,T₂<:SMatrix{3,3,T,9},T₃<:SVector{3,T}}
    Threads.@threads for i ∈ eachindex(PC)
        @inbounds PC[i] = R * PC[i] .+ t
    end
end
function transform_PC!(PC::Vector{SVector{2,T}}, R::T₂, t::T₃) where {T<:Float,T₂<:SMatrix{2,2,T,4},T₃<:SVector{2,T}}
    Threads.@threads for i ∈ eachindex(PC)
        @inbounds PC[i] = R * PC[i] .+ t
    end
end

"""
	transform_PC!(PC,T)
Inplace transformation of a Point Cloud `PC` under transformation `T`

See also [`transform_PC`](@ref)
"""
function transform_PC!(PC::Vector{SVector{3,T₁}}, T::T₂) where {T₁<:Float,T₂<:SMatrix{4,4,T₁,16}}
    R, t = T2rt(T)
    Threads.@threads for i ∈ eachindex(PC)
        @inbounds PC[i] = R * PC[i] .+ t
    end
end

function transform_PC!(PC::Vector{SVector{2,T₁}}, T::T₂) where {T₁<:Float,T₂<:SMatrix{3,3,T₁,9}}
    R, t = T2rt(T)
    Threads.@threads for i ∈ eachindex(PC)
        @inbounds PC[i] = R * PC[i] .+ t
    end
end

"""
	P2P_ICP(source,target, w)
Point to Point ICP with SVD. **Points in source and points in target are listed correspoindingly, given weight `w`**
"""
function P2P_ICP(X::Vector{SVector{D,T}}, Y::Vector{SVector{D,T}}, w::Vector{T}=ones(T, size(X, 1))) where {T<:Float,D}
    wₙ = w / sum(w)
    meanₛ = sum(X .* wₙ)
    X = X .- [meanₛ]
    meanₜ = sum(Y .* wₙ)
    Y = Y .- [meanₜ]
    Σ = reduce(hcat, X) * reduce(hcat, Y .* wₙ)'
    F = svd(Σ)
    U = SMatrix{D,D,T,D * D}(F.U)
    V = SMatrix{D,D,T,D * D}(F.V)
    if det(U) * det(V) < 0
        s = ones(T, D)
        s[end] = -s[end]
        S = SMatrix{D,D,T,D * D}(diagm(s))
        R = V * S * U'
    else
        R = V * U'
    end
    t = meanₜ - R * meanₛ
    X = X .+ [meanₛ]
    Y = Y .+ [meanₜ]
    return rt2T(R, t)
end

"""
	get_energy(W,ν₁,f)
get point cloud error energy given error `W`, parameter `ν₁`, with function specified by `f`
"""
function get_energy(W, ν₁, f)
    if f == "tukey"
        return tukey_energy(W, ν₁)
    elseif f == "fair"
        return fair_energy(W, ν₁)
    elseif f == "log"
        return logistic_energy(W, ν₁)
    elseif f == "trimmed"
        return trimmed_energy(W, ν₁)
    elseif f == "welsch"
        return welsch_energy(W, ν₁)
    elseif f == "auto_welsch"
        return autowelsch_energy(W, ν₁)
    elseif f == "uniform"
        return uniform_energy(W)
    else
        return uniform_energy(W)
    end
end

"""
	robust_weight!(W,ν₁,f)
get point cloud robust weight given error `W`, parameter `ν₁`, with function specified by `f`
"""
function robust_weight!(W, ν₁, f)
    if f == "tukey"
        W .= tukey_weight(W, ν₁)
    elseif f == "fair"
        W .= fair_weight(W, ν₁)
    elseif f == "log"
        W .= logistic_weight(W, ν₁)
    elseif f == "trimmed"
        W .= trimmed_weight(W, ν₁)
    elseif f == "welsch"
        W .= welsch_weight(W, ν₁)
    elseif f == "auto_welsch"
        W .= autowelsch_weight(W, ν₁)
    elseif f == "uniform"
        W .= uniform_weight(W)
    else
        W .= uniform_weight(W)
    end
end

begin
    uniform_weight(r::Vector{T}) where {T} = @inline ones(T, size(r, 1))
    pnorm_weight(r::Vector{T}, p::T, reg=T(1e-16)) where {T} = @inline p ./ (r .^ (2 - p) .+ reg)
    tukey_weight(r::Vector{T}, p::T) where {T} = @inline (T(1.0) .- (r ./ p) .^ (T(2.0))) .^ T(2.0)
    fair_weight(r::Vector{T}, p::T) where {T} = @inline T(1.0) ./ (T(1.0) .+ r ./ p)
    logistic_weight(r::Vector{T}, p::T) where {T} = @inline (p ./ r) .* tanh.(r ./ p)
    welsch_weight(r::Vector{T}, p::T) where {T} = @inline exp.(-(r .* r) ./ (2 * p * p))
    autowelsch_weight(r::Vector{T}, p::T) where {T} = welsch_weight(r, p * median(r) / T(sqrt(2.0) * 2.3))
    function trimmed_weight(r::Vector{T}, p::T) where {T}
        return error
    end
end

begin
    function tukey_energy(r::Vector{T}, p::T) where {T}
        r = (1 .- (r ./ p) .^ 2) .^ 2
        r[r.>p] = T(0.0)
        return r
    end
    function trimmed_energy(r::Vector{T}, p::T) where {T}
        return zeros(T, size(r, 1))
        ## TODO: finish trimmed_energy function
    end
    fair_energy(r, p) = @inline sum(r .^ 2) ./ (1 .+ r ./ p)
    logistic_energy(r, p) = @inline sum(r .^ 2 .* (p ./ r) * tanh.(r ./ p))
    welsch_energy(r::Vector{T}, p::T) where {T} = @inline sum(T(1.0) .- exp.(-r .^ 2 ./ (T(2.0) .* p .* p)))
    autowelsch_energy(r::Vector{T}, p::T) where {T} = welsch_energy(r, T(0.5))
    uniform_energy(r::Vector{T}) where {T} = @inline ones(T, size(r, 1))
end

function FindKnearestMed(P::Vector{SVector{D,T}}, Tree, k) where {T,D}
    n = size(P, 1)
    Xnearest = Vector{T}(undef, n)
    Threads.@threads for i = 1:n
        idxs, dists = knn(Tree, P[i], k, true)
        Xnearest[i] = median(dists)
    end
    return sqrt(median(Xnearest))
end
