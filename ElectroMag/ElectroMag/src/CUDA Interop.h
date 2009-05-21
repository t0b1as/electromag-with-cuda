#pragma once

#include "Data Structures.h"
#include "Electrostatics.h"
#include "./../../GPGPU Segment/src/GPU manager.h"

int CUDA_QueryActive();

// Include the GPGPU library if compiling external project
#ifndef __CUDACC__
#pragma comment(lib, "GPGPU.lib")
#endif


int CalcField(Array<Vector3<float> >& fieldLines, Array<pointCharge<float> >& pointCharges,
			  size_t n, float resolution,  perfPacket& perfData, bool useCurvature = false);
int CalcField(Array<Vector3<double> >& fieldLines, Array<pointCharge<double> >& pointCharges,
			  size_t n, double resolution,  perfPacket& perfData, bool useCurvature = false);

// This allows us to readibly keep track of timing information within wrappers
enum CalcField_timingSteps{xySize, zSize, devMalloc, hostMalloc, xyHtoH, xyHtoD, zHtoH, zHtoD, xyDtoH, xyHtoHb, zDtoH, zHtoHb, mFree, timingSize};
// and disable the idiotic warning: "warning #2158: enum qualified name is nonstandard"
#pragma warning(disable:2158)
