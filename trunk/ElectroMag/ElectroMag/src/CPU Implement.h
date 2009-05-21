#pragma once
#include "Data Structures.h"
#include "Electrostatics.h"

template<class T>
int CalcField_CPU(Array<Vector3<T> >& fieldLines, Array<pointCharge<T> >& pointCharges,
			  const __int64 n, T resolution, perfPacket& perfData, bool useCurvature = false);

