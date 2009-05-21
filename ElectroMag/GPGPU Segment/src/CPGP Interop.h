#pragma once
#include "Data Structures.h"
#include "Config.h"

template<class T>
/*inline*/ cudaError_t CalcField_core(Vec3SOA<T> &fieldLines, unsigned int steps, unsigned int lines,
							  unsigned int xyPitchOffset, unsigned int zPitchOffset,
							  pointCharge<T> * pointCharges, unsigned int points,
							  T resolution, bool useMT, bool useCurvature);


