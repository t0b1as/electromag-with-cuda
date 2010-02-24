/***********************************************************************************************
Copyright (C) 2009 - Alexandru Gagniuc - <http:\\g-tech.homeserver.com\HPC.htm>
 * This file is part of ElectroMag.

    ElectroMag is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    ElectroMag is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with ElectroMag.  If not, see <http://www.gnu.org/licenses/>.
***********************************************************************************************/


#ifndef _VECTOR_H
#define _VECTOR_H
#include "CUDA Definitions.h"


template <class T>
struct Vector2
{
	T x, y;
};

#ifndef __CUDACC__
template <class T>
struct Vector3
{
	T x, y, z;
};
#else
template <class T>
struct Vector3
{
	union
	{
		Vector2<T> xy;
		struct
		{
			T x, y;
		};
	};
	T z;
};
/* // Warning: the specialization causes kernel execution to terminate with unknown error 
// Explicit structure alignment
template<>
struct __align__(8) Vector2<float>
{
	float x, y;
};
template<>
struct __align__(16) Vector2<double>
{
	double x, y;
};
*/
#endif

template<class T>
struct Vec3SOA
{
	Vector2<T>* xyInterleaved;
	T* z;
};

/*=============================================================================
C++ style vector functions and operators
=============================================================================*/
template <class T>
__device__ inline void operator += (Vector3<T> &rhs, /*const*/ Vector3<T> b)
{
	rhs.x += b.x; rhs.y += b.y; rhs.z += b.z;		// 3 FLOPs
};

template <class T>
__device__ inline Vector3<T> operator + (const Vector3<T> A, const Vector3<T> B)
{
	return vec3Add(A, B);			// 3 FLOPs
};

template <class T>
__device__ inline Vector3<T> operator - (const Vector3<T> A, const Vector3<T> B)
{
	return vec3Sub(A, B);			// 3 FLOPs
};

template <class T>
__device__ inline Vector3<T> operator * (Vector3<T> vec, T scalar)
{
	return vec3Mul(vec, scalar);	// 3 FLOPs
};

template <class T>
__device__ inline Vector3<T> operator / (Vector3<T> vec, T scalar)
{
	return vec3Div(vec, scalar);	// 3 FLOPs
};

/*=============================================================================
C style vector functions
=============================================================================*/

template<class T>
__device__ inline Vector3<T> vec3(Vector3<T> head, Vector3<T> tail)
{
	Vector3<T> result = {head.x-tail.x, head.y-tail.y, head.z-tail.z};
	return result;	// 3 FLOPs
};
template <class T>
__device__ inline void vec3Addto(Vector3<T>& rhs, const Vector3<T> b)
{
	rhs.x += b.x; rhs.y += b.y; rhs.z += b.z;		// 3 FLOPs
};
template <class T>
__device__ inline Vector3<T> vec3Add(Vector3<T> A, Vector3<T> B)
{
	Vector3<T> result;
	result.x = A.x + B.x;
	result.y = A.y + B.y;
	result.z = A.z + B.z;
	return result;	// 3 FLOPs
};
template <class T>
__device__ inline Vector3<T> vec3Sub(Vector3<T> A, Vector3<T> B)
{
	Vector3<T> result;
	result.x = A.x - B.x;
	result.y = A.y - B.y;
	result.z = A.z - B.z;
	return result;	// 3 FLOPs
};
template <class T>
__device__ inline Vector3<T> vec3Mul(Vector3<T> vec, T scalar)
{
	Vector3<T> result;
	result.x = vec.x*scalar;
	result.y = vec.y*scalar;
	result.z = vec.z*scalar;
	return result;	// 3 FLOPs
};
template <class T>
__device__ inline Vector3<T> vec3Div(Vector3<T> vec, T scalar)
{
	Vector3<T> result;
	result.x = vec.x/scalar;
	result.y = vec.y/scalar;
	result.z = vec.z/scalar;
	return result;	// 3 FLOPs
};

template <class T>
__device__ inline Vector3<T> vec3Unit(Vector3<T> vec)
{
	T len = vec3Len(vec);									// 6 FLOPs
	Vector3<T> result = {vec.x/len, vec.y/len, vec.z/len};	// 3 FLOPs
	return result;										// Total: 9 FLOPs
};
// Saves 2 FLOPs when taking the unit of a vector and multiplying it by a scalar
template <class T>
__device__ inline Vector3<T> vec3SetLen(Vector3<T> vec, T scalarLen)
{
	T len = vec3Len(vec);									// 6 FLOPs
	scalarLen /= len;										// 1 FLOP
	return vec3Mul(vec, scalarLen);							// 3 FLOPs
	// Total: 10 FLOPs
};
// Same as above, but works by dividing by len rather than multiplying
template <class T>
__device__ inline Vector3<T> vec3SetInvLen(Vector3<T> vec, T scalarInvLen)
{
	T len = vec3Len(vec);										// 6 FLOPs
	scalarInvLen *= len;										// 1 FLOP
	return vec3Div(vec, scalarInvLen);							// 3 FLOPs
	// Total: 10 FLOPs
};

template <class T>
__device__ inline T vec3LenSq(Vector3<T> vec)
{
	return (vec.x*vec.x + vec.y*vec.y + vec.z*vec.z);		// 5 FLOPs
};
template <class T>
__device__ inline T vec3Len(Vector3<T> vec)
{
	return sqrt(vec3LenSq(vec));							// 6 FLOPs
};

template <class T>
__device__ inline Vector3<T> vec3Cross(const Vector3<T> index, const Vector3<T>middle)
{
	Vector3<T> result;
	result.x = index.y * middle.z - index.z * middle.y;		// 3 FLOPs
	result.y = index.z * middle.x - index.x * middle.z;		// 3 FLOPs
	result.z = index.x * middle.y - index.y * middle.x;		// 3 FLOPs
	return result;							// Total: 9 FLOPs
};


#endif//_VECTOR_H



