#pragma once
#include "Electrostatics.h"

template <class T>
struct __align__(16) dynamicPointCharge
{
	pointCharge<T> staticProperties;
	Vector3<T> velocity;
	T mass;
};

#ifdef __CUDACC__
// Alignment not specified yet
//template __align__(16) dynamicPointCharge<float>
#endif
