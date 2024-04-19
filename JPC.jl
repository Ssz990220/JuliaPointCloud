module JPC
using Reexport
@reexport using StaticArrays, Makie, GLMakie
@reexport using Random, LinearAlgebra, Statistics, StaticArrays
@reexport using NearestNeighbors
@reexport using Rotations
@reexport using MeshIO, FileIO, GeometryBasics

const Float = Union{Float32,Float64}
include("./src/alge.jl")
export T2rt, rt2T, MatrixLog3, MatrixLog6, wrapto1, se32vec, vec2se3

include("./src/PC_Visualization.jl")
export visualize, visualize!

include("./src/utils.jl")
export load_PC, PC2SVector, Point2PC, SVector2PC, transform_PC, transform_PC!, downSample_PC

include("./src/FRICP.jl")
export FICP_P2P

end