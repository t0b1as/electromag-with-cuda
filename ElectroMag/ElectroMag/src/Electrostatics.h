#pragma once
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
		lenSq / (T)sqrt(lenSq) );						// 4 FLOP (1 sqrt + 3 mul-div)
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
