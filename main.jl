include("FRICP.jl")
using .FRICP
begin
	PC, N = load_PC("./Assets/source.ply")
	source = PC;
	PCₜ, Nₜ = load_PC("./Assets/target.ply")
	target = PCₜ
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

FICP_P2P(source,target,par)
