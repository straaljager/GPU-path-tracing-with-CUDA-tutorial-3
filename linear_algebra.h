/*
*  CUDA based triangle mesh path tracer using BVH acceleration by Sam lapere, 2016
*  BVH implementation based on real-time CUDA ray tracer by Thanassis Tsiodras,
*  http://users.softlab.ntua.gr/~ttsiod/cudarenderer-BVH.html
*
*  This program is free software; you can redistribute it and/or modify
*  it under the terms of the GNU General Public License as published by
*  the Free Software Foundation; either version 2 of the License, or
*  (at your option) any later version.
*
*  This program is distributed in the hope that it will be useful,
*  but WITHOUT ANY WARRANTY; without even the implied warranty of
*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*  GNU General Public License for more details.
*
*  You should have received a copy of the GNU General Public License
*  along with this program; if not, write to the Free Software
*  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
*/
#ifndef __LINEAR_ALGEBRA_H_
#define __LINEAR_ALGEBRA_H_

#include <cuda_runtime.h> // for __host__  __device__
#include <math.h>

struct Vector3Df
{
	union {
		struct { float x, y, z; };
		float _v[3];
	};

	__host__ __device__ Vector3Df(float _x = 0, float _y = 0, float _z = 0) : x(_x), y(_y), z(_z) {}
	__host__ __device__ Vector3Df(const Vector3Df& v) : x(v.x), y(v.y), z(v.z) {}
	__host__ __device__ Vector3Df(const float4& v) : x(v.x), y(v.y), z(v.z) {}
	inline __host__ __device__ float length(){ return sqrtf(x*x + y*y + z*z); }
	// sometimes we dont need the sqrt, we are just comparing one length with another
	inline __host__ __device__ float lengthsq(){ return x*x + y*y + z*z; }
	inline __host__ __device__ void normalize(){ float norm = sqrtf(x*x + y*y + z*z); x /= norm; y /= norm; z /= norm; }
	inline __host__ __device__ Vector3Df& operator+=(const Vector3Df& v){ x += v.x; y += v.y; z += v.z; return *this; }
	inline __host__ __device__ Vector3Df& operator-=(const Vector3Df& v){ x -= v.x; y -= v.y; z -= v.z; return *this; }
	inline __host__ __device__ Vector3Df& operator*=(const float& a){ x *= a; y *= a; z *= a; return *this; }
	inline __host__ __device__ Vector3Df& operator*=(const Vector3Df& v){ x *= v.x; y *= v.y; z *= v.z; return *this; }
	inline __host__ __device__ Vector3Df operator*(float a) const{ return Vector3Df(x*a, y*a, z*a); }
	inline __host__ __device__ Vector3Df operator/(float a) const{ return Vector3Df(x/a, y/a, z/a); }
	inline __host__ __device__ Vector3Df operator*(const Vector3Df& v) const{ return Vector3Df(x * v.x, y * v.y, z * v.z); }
	inline __host__ __device__ Vector3Df operator+(const Vector3Df& v) const{ return Vector3Df(x + v.x, y + v.y, z + v.z); }
	inline __host__ __device__ Vector3Df operator-(const Vector3Df& v) const{ return Vector3Df(x - v.x, y - v.y, z - v.z); }
	inline __host__ __device__ Vector3Df& operator/=(const float& a){ x /= a; y /= a; z /= a; return *this; }
	inline __host__ __device__ bool operator!=(const Vector3Df& v){ return x != v.x || y != v.y || z != v.z; }
};


inline __host__ __device__ Vector3Df min3(const Vector3Df& v1, const Vector3Df& v2){ return Vector3Df(v1.x < v2.x ? v1.x : v2.x, v1.y < v2.y ? v1.y : v2.y, v1.z < v2.z ? v1.z : v2.z); }
inline __host__ __device__ Vector3Df max3(const Vector3Df& v1, const Vector3Df& v2){ return Vector3Df(v1.x > v2.x ? v1.x : v2.x, v1.y > v2.y ? v1.y : v2.y, v1.z > v2.z ? v1.z : v2.z); }
inline __host__ __device__ Vector3Df cross(const Vector3Df& v1, const Vector3Df& v2){ return Vector3Df(v1.y*v2.z - v1.z*v2.y, v1.z*v2.x - v1.x*v2.z, v1.x*v2.y - v1.y*v2.x); }
inline __host__ __device__ float dot(const Vector3Df& v1, const Vector3Df& v2){ return v1.x*v2.x + v1.y*v2.y + v1.z*v2.z; }
inline __host__ __device__ float dot(const Vector3Df& v1, const float4& v2){ return v1.x*v2.x + v1.y*v2.y + v1.z*v2.z; }
inline __host__ __device__ float dot(const float4& v1, const Vector3Df& v2){ return v1.x*v2.x + v1.y*v2.y + v1.z*v2.z; }
inline __host__ __device__ float distancesq(const Vector3Df& v1, const Vector3Df& v2){ return (v1.x - v2.x)*(v1.x - v2.x) + (v1.y - v2.y)*(v1.y - v2.y) + (v1.z - v2.z)*(v1.z - v2.z); }
inline __host__ __device__ float distance(const Vector3Df& v1, const Vector3Df& v2){ return sqrtf((v1.x - v2.x)*(v1.x - v2.x) + (v1.y - v2.y)*(v1.y - v2.y) + (v1.z - v2.z)*(v1.z - v2.z)); }

#endif
