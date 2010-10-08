/***********************************************************************************************
Copyright (C) 2010 - Alexandru Gagniuc - <http:\\g-tech.homeserver.com\HPC.htm>
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


#ifndef _ELECTROSTATICS_H
#define _ELECTROSTATICS_H

#include "CUDA Definitions.h"
#include "Vector.h"


namespace electro
{

#define electro_k  8.987551787E9


template <class T>
struct pointCharge
{
	Vector3<T> position;
	T magnitude;
};

#ifdef __CUDACC__
// Allows point charges to be read and written as float4 type
template <>
struct __align__(16) pointCharge<float>
{
	union
	{
		float4 ftype;
		struct
		{
			Vector3<float> position;
			float magnitude;
		};
	};
};

template<>
struct __align__(16) pointCharge<double>
{
	Vector3<double> position;
	double magnitude;
};
#endif

// We need this for wierd types
template <class T>
inline __device__ Vector3<T> PartField(pointCharge<T> charge, Vector3<T> point, T electroK)
{
	Vector3<T> r = vec3(point, charge.position);		// 3 FLOP
	T lenSq = vec3LenSq(r);								// 5 FLOP
	return r * (T)electroK * charge.magnitude /	// 3 FLOP (vecMul)
		(lenSq * (T)sqrt(lenSq));						// 4 FLOP (1 sqrt + 3 mul,div)
	// NOTE: instead of dividing by lenSq and then sqrt(lenSq), we only divide once by their product
	// Since we have one division and one multiplication, this should be more efficient due to the
	// Fact that most architectures perform multiplication way faster than division. Also on some
	// GPU architectures this yields a more precise result.
}						// Total: 15 FLOP

template <class T>
inline __device__ Vector3<T> PartField(pointCharge<T> charge, Vector3<T> point)
{
	Vector3<T> r = vec3(point, charge.position);		// 3 FLOP
	T lenSq = vec3LenSq(r);								// 5 FLOP
	return vec3Mul(r, (T)electro_k * charge.magnitude /	// 3 FLOP (vecMul)
		(lenSq * (T)sqrt(lenSq)) );						// 4 FLOP (1 sqrt + 3 mul,div)
}	

#if defined(__CUDACC__)
// Yet another CUDA optimization:
// Square root is performed by raciprocal square root followed by reciprocal, which are two expensive operations
// Since we divide by the square root of lenSq, it makes insanely more sense to multiply by the reciprocal square root of lenSq
// since division with lenSq will be executed asa a reciprocal and multiplication, we can multiply by the reciprocal of lenSq
// NOTE: This might change with future architectures, so keep an eye on the programming guide, and see how Fermi performs
template <>
__device__ Vector3<float> PartField(pointCharge<float> charge, Vector3<float> point)
{
	Vector3<float> r = vec3(point, charge.position);		// 3 FLOP
	float lenSq = vec3LenSq(r);								// 5 FLOP
	return vec3Mul(r, (float)electro_k * charge.magnitude *	// 3 FLOP (vecMul)
		rsqrtf(lenSq) / lenSq );							// 4 FLOP (1 sqrt + 3 mul,div)
};
template <>
__device__ Vector3<double> PartField(pointCharge<double> charge, Vector3<double> point)
{
	Vector3<double> r = vec3(point, charge.position);			// 3 FLOP
	double lenSq = vec3LenSq(r);								// 5 FLOP
	return vec3Mul(r, (double)electro_k * charge.magnitude *	// 3 FLOP (vecMul)
		rsqrt(lenSq) / lenSq );								// 4 FLOP (1 sqrt + 3 mul,div)
};
#endif

#define electroPartFieldFLOP 15

////////////////////////////////////////////////////////////////////////////////////////////////
///\brief  Operates on the inverse quare vector to give the magnetic field
///
////////////////////////////////////////////////////////////////////////////////////////////////
template <class T>
inline __device__ Vector3<T> PartFieldOp(
		T qSrc,
		Vector3<T> rInvSq
		)
{
	return (T)electro_k * qSrc * rInvSq;	// 4 FLOP ( 1 mul, 3 vecMul)
					// Total: 5 FLOP
}
#define electroPartFieldOpFLOP 5

// Returns the partial field vector without multiplying by electro_k to save one FLOP
template <class T>
inline __device__ Vector3<T> PartFieldVec(pointCharge<T> charge, Vector3<T> point)
{
	Vector3<T> r = vec3(point, charge.position);		// 3 FLOP
	T lenSq = vec3LenSq(r);								// 5 FLOP
	return vec3Mul(r, charge.magnitude /				// 3 FLOP (vecMul)
		(lenSq * (T)sqrt(lenSq)) );						// 3 FLOP (1 sqrt + 2 mul-div)
}						// Total: 14 FLOP
#define electroPartFieldVecFLOP 14

#if defined(__CUDACC__)
// Yet another CUDA optimization:
// Square root is performed by raciprocal square root followed by reciprocal, which are two expensive operations
// Since we divide by the square root of lenSq, it makes insanely more sense to multiply by the reciprocal square root of lenSq
// since division with lenSq will be executed asa a reciprocal and multiplication, we can multiply by the reciprocal of lenSq
// NOTE: This might change with future architectures, so keep an eye on the programming guide, and see how Fermi performs
template <>
__device__ Vector3<float> PartFieldVec(pointCharge<float> charge, Vector3<float> point)
{
	Vector3<float> r = vec3(point, charge.position);		// 3 FLOP
	float lenSq = vec3LenSq(r);								// 5 FLOP
	return vec3Mul(r, charge.magnitude *					// 3 FLOP (vecMul)
		rsqrtf(lenSq) / lenSq );							// 4 FLOP (1 sqrt + 3 mul,div)
};
template <>
__device__ Vector3<double> PartFieldVec(pointCharge<double> charge, Vector3<double> point)
{
	Vector3<double> r = vec3(point, charge.position);		// 3 FLOP
	double lenSq = vec3LenSq(r);							// 5 FLOP
	return vec3Mul(r, charge.magnitude *					// 3 FLOP (vecMul)
		rsqrt(lenSq) / lenSq );								// 4 FLOP (1 sqrt + 3 mul,div)
};
#endif

}//namespace electro

#endif// _ELECTROSTATICS_H
