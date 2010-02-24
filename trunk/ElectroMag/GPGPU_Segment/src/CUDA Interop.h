/***********************************************************************************************
Copyright (C) 2009-2010 - Alexandru Gagniuc - <http:\\g-tech.homeserver.com\HPC.htm>
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

#ifndef _CUDA_INTEROP_H
#define _CUDA_INTEROP_H

#include "Data Structures.h"
#include "Electrostatics.h"
#include "CUDA Manager.h"

// Include the GPGPU library if compiling external project
#if !defined(__CUDACC__) && ( defined(_WIN32) || defined(_WIN64) )
#pragma comment(lib, "GPGPU.lib")
#endif


int CalcField(Array<Vector3<float> >& fieldLines, Array<pointCharge<float> >& pointCharges,
			  size_t n, float resolution,  perfPacket& perfData, bool useCurvature = false);
int CalcField(Array<Vector3<double> >& fieldLines, Array<pointCharge<double> >& pointCharges,
			  size_t n, double resolution,  perfPacket& perfData, bool useCurvature = false);

// This allows us to readibly keep track of timing information within wrappers
// NOTE: any change here, and all sources should be recompiled to be on the safe side
enum CalcField_timingSteps
{
	xySize,			///< Size in bytes of xy components
	zSize,			///< Size in bytes of z components
	kernelLoad,		///< Time for loading the kernel module
	kernelExec,		///< Time for kernel execution
	resAlloc,		///< Time for allocating resources, including device and host memory
	xyHtoH,			///< Time for transferring xy componens from main array to page-locked memory
	xyHtoD,			///< Time for transferring xy componens from page-locked memory to device
	zHtoH,			///< Time for transferring z  componens from main array to page-locked memory
	zHtoD,			///< Time for transferring z  componens from page-locked memory to device
	xyDtoH,			///< Time for transferring xy componens from device to page-locked memory
	xyHtoHb,		///< Time for transferring xy componens from page-locked memory to main array
	zDtoH,			///< Time for transferring z  componens from device to page-locked memory
	zHtoHb,			///< Time for transferring z  componens from page-locked memory to main array
	mFree,			///< Time for freeing page-locked and device memory
	timingSize		///< Total number of timing/performance parameters
};

#endif//_CUDA_INTEROP_H
