module JPC
    using Reexport
    @Reexport using StaticArrays, Makie, GLMakie
    @reexport using Random, LinearAlgebra, Statistics, StaticArrays
    @reexport using NearestNeighbors
    @reexport using Flux3D
    @reexport using Rotations
    @reexport using MeshIO,FileIO,GeometryBasics

    
    include("./Visualization/PC_Visualization.jl")
    export visualize, visualize!

    
    include("./Scripts/utils.jl")
    export load_PC, PC2SVector, Point2PC, SVector2PC

    include("./Scripts/FRICP.jl")
    export FICP_P2P
    
end