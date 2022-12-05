using Makie, GLMakie

"""
    visualize(pcloud::PointCloud; kwargs...)

Visualize PointCloud `pcloud` at `index`.

Dimension of points in PointCloud `pcloud` must be 3.

### Optional Arguments:
- color (Symbol)       - Color of the marker, default `:blue`
- markersize (Number)  - Size of the marker, default `npoints(pcloud)/10000`

"""
function visualize(p::PointCloud, index::Number = 1; kwargs...)
    points = cpu(p[index])
    size(points, 1) == 3 || error("dimension of points in PointCloud must be 3.")

    kwargs = convert(Dict{Symbol,Any}, kwargs)
    get!(kwargs, :color, :lightgreen)
    get!(kwargs, :markersize, npoints(p) / 10000)

    meshscatter(points[1, :], points[2, :], points[3, :]; kwargs...)
end

function visualize!(axis3::Makie.Axis3, p::PointCloud, index::Number = 1; kwargs...)
    points = cpu(p[index])
    size(points, 1) == 3 || error("dimension of points in PointCloud must be 3.")

    kwargs = convert(Dict{Symbol,Any}, kwargs)
    get!(kwargs, :color, :lightgreen)
    get!(kwargs, :markersize, npoints(p) / 10000)

    meshscatter!(axis3, points[1, :], points[2, :], points[3, :]; kwargs...)
end

function visualize(PC::Vector{SMatrix};kwargs...)
    size(PC[1],1) == 3 || error("dimension of points in PointCloud must be 3.")
    kwargs = convert(Dict{Symbol,Any},kwargs)
    get!(kwargs, :color, :lightgreen)
    get!(kwargs, :markersize, npoints(p) / 10000)
    meshscatter()

end