
"""
	rt2T(R,t)
Combine a rotation matrix `R` and a translation vector `t` to a homogenous matrix `T`
"""
function rt2T(R::StaticArray{Tuple{D,D},T,2},t::SVector{D,T}) where {D,T<:Float}
	row = zeros(1,D+1); row[1,D+1] = T(1.0);
	row = SMatrix{1,D+1,T,D+1}(row);
	return vcat(hcat(R,t),row)
end

function rt2T(R::StaticArray{Tuple{3, 3},T,2},t::SVector{3,T}) where {T<:Float}
	row = SMatrix{1,D+1,T,D+1}(0.0,0.0,0.0,1.0);
	return vcat(hcat(R,t),row)
end


function rt2T(R::StaticArray{Tuple{2, 2},T,2},t::SVector{2,T}) where {T<:Float}
	row = SMatrix{1,D+1,T,D+1}(0.0,0.0,1.0);
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

function MatrixLog3(R::SMatrix{D,D,T,D₁}) where {T<:Float,D,D₁}
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

function MatrixLog6(T::SMatrix{D,D,T₁,D₁}) where {T₁,D,D₁}
	R, p = T2rt(T);
	omgmat = MatrixLog3(R);
	if isequal(omgmat, zeros(T₁,3,3))
		R = @SMatrix zeros(T₁,3,3)
		expmat = vcat(hcat(R,T[SOneTo(3), 4]) ,@SMatrix zeros(D,1,4));
	else
		theta = acos(wrapto1((tr(R) - 1.0f0) / 2.0f0));
		expmat = vcat(hcat(omgmat,(I - omgmat ./ 2.0f0 + (1.0f0 / theta - cot(theta / 2.0f0) / 2.0f0) * omgmat * omgmat / theta) * p),@SMatrix zeros(T₁,1,4));
	end
	return expmat
end

function wrapto1(a::T) where {T<:Float}
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