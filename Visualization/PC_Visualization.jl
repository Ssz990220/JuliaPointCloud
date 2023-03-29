"""

    visualize(PC::Vector{SMatrix{D,1,T,D}};kwargs...)

Visualize PointCloud `PC`

Dimension of points in PointCloud `PC` must be 3.

### Return Values:
- fig (Figure)          - Created Figure Canvas
- axis(Axis3)           - Default Plot Axis
- P(Observable)         - Mutable Point Cloud Handler
- obj(MeshscatterObj)   - MeshScatter Container
"""

function visualize(PC::Vector{SMatrix{D,1,T,D}};kwargs...) where {T,D}
    @assert D==3
    n = size(PC,1)

    ## axis labels
    kwargs = convert(Dict{Symbol,Any},kwargs)
    get!(kwargs, :xlabel,"X"); get!(kwargs, :ylabel,"Y"); get!(kwargs, :zlabel, "Z")

    ## MarkerSize
    pc = zeros(T,D,n)
    @inbounds for i = 1:n
        pc[:,i] = PC[i]
    end
    markersize = (maximum(maximum(pc,dims = 2) .- minimum(pc,dims=2)))/1e3

    ## Observables
    if T == Float32
        P = Observable(Vector{Point3f0}(undef,0))
    else
        P = Observable(Vector{Point3}(undef,0))
    end
    downsample = Int(n > 1e5 ? floor(n/1e5) : 1)
    P[] = push!(P[],PC[1:downsample:n]...)

    ## Color Map
    z = @lift([$P[i][3] for i ∈ eachindex(P[])]); zₘₐₓ = @lift maximum($z); zₘᵢₙ = @lift minimum($z); 
    val = @lift ($z.-$zₘᵢₙ)/($zₘₐₓ-$zₘᵢₙ);

    # axis
    fig = Figure()
    axis = Axis3(fig[1, 1],
        perspectiveness = 0.5,
        # azimuth = 2.19,
        # elevation = 0.57,
        xlabel = kwargs[:xlabel],
        ylabel = kwargs[:ylabel],
        zlabel = kwargs[:zlabel],
        aspect = (1, 1, 1)
        )
    
    theme = get!(kwargs,:theme,"dark") == "dark" ? theme_dark() : theme_light();

    MSizeIdx = Observable(markersize)
    MSize = @lift (markersize / 5^$MSizeIdx)
    with_theme(theme) do
        plotobj = meshscatter!(axis, P, color=val,colormap=:viridis,markersize = MSize)
        sl = Slider(fig[end+1, 1], range = -5:0.5:5, startvalue = 0)
        connect!(MSizeIdx, sl.value)
        return fig, axis, P, plotobj
    end
end

"""

    visualize!(axis,plotobj,PC::Vector{SMatrix{D,1,T,D}};kwargs...)

Visualize PointCloud `PC` on `axis` with plot variable given by `plotobj`

Dimension of points in PointCloud `PC` must be 3.

### Return Values:
- P(Observable)         - Mutable Point Cloud Handler

"""

function visualize!(axis,plotobj,PC::Vector{SMatrix{D,1,T,D}};kwargs...) where {T,D}
    @assert D==3
    n = size(PC,1)

    ## Observables
    if T == Float32
        P = Observable(Vector{Point3f0}(undef,0))
    else
        P = Observable(Vector{Point3}(undef,0))
    end
    downsample = Int(n > 1e5 ? floor(n/1e5) : 1)
    P[] = push!(P[],PC[1:downsample:n]...)

    ## Color Map
    z = @lift([$P[i][3] for i ∈ eachindex(P[])]); zₘₐₓ = @lift maximum($z); zₘᵢₙ = @lift minimum($z); 
    val = @lift ($z.-$zₘᵢₙ)/($zₘₐₓ-$zₘᵢₙ);
    meshscatter!(axis,P;color=val,markersize = plotobj.markersize)
    return P
end