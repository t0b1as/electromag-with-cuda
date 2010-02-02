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

template <class T>
inline __device__ Vector3<T> electroPartField(pointCharge<T> charge, Vector3<T> point)
{
	Vector3<T> r = vec3(point, charge.position);		// 3 FLOP
	T lenSq = vec3LenSq(r);								// 5 FLOP
	return vec3Mul(r, (T)electro_k * charge.magnitude /	// 3 FLOP (vecMul)
		(lenSq * (T)sqrt(lenSq)) );						// 4 FLOP (1 sqrt + 3 mul-div)
};						// Total: 15 FLOP
#define electroPartFieldFLOP 15

// Returns the partial field vector without multiplying by electro_k to save one FLOP
template <class T>
inline __device__ Vector3<T> electroPartFieldVec(pointCharge<T> charge, Vector3<T> point)
{
	Vector3<T> r = vec3(point, charge.position);		// 3 FLOP
	T lenSq = vec3LenSq(r);								// 5 FLOP
	return vec3Mul(r, charge.magnitude /				// 3 FLOP (vecMul)
		lenSq / (T)sqrt(lenSq) );						// 3 FLOP (1 sqrt + 2 mul-div)
};						// Total: 14 FLOP
#define electroPartFieldVecFLOP 14

#endif// _ELECTROSTATICS_H
