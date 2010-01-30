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
#pragma once

#include"cuda_drvapi_dynlink.h"


//Selects the appropriate kernel
// Selection must be done in specialized functions, and the generic function should return an error
template<class T>
CUresult CalcField_selectKernel(CUmodule multistepMod, CUmodule singlestepMod,
                        CUfunction *multistepKernel, CUfunction *singlestepKernel,
                        bool useMT, bool useCurvature)
{
    // The generic template does nothing.
	// There is no optimization for generic types
	return CUDA_ERROR_INVALID_IMAGE;
}

#include <iostream>

struct GPUkernels
{
	CUfunction singlestepKernel;
	CUfunction multistepKernel;
};



#define DEBUG_CUDA_CALL(call) errCode = call;\
	if(errCode != CUDA_SUCCESS)\
{std::cerr<<" Failed at "<<__FILE__<<" line "<<__LINE__<<" in function "<<__FUNCTION__<<" with code: "<<errCode<<std::endl;\
return errCode;}

#define CUDA_CPP_CALL(call) errCode = call;\
	if(errCode != CUDA_SUCCESS)\
	{std::cerr<<" Failed at "<<__FILE__<<" line "<<__LINE__<<" in function "<<__FUNCTION__<<" with code: "<<errCode<<std::endl;}\

#define DEBUG_CUDA_KERNEL_SYNC(call) errCode = call;\
	if(errCode != CUDA_SUCCESS)\
	{\
		std::cerr<<" Failed kernel sync at line line "<<__LINE__<<" with code: "<<errCode<<std::endl;\
		std::cerr<<" Current unprocessed fieldIndex "<<i<<std::endl;\
		return errCode;\
	}

template<>
CUresult CalcField_selectKernel<float>(CUmodule multistepMod, CUmodule singlestepMod,
                        CUfunction *multistepKernel, CUfunction *singlestepKernel,
                        bool useMT, bool useCurvature)
{
	CUresult errCode;
    if(useCurvature)	// Curvature computation is only available in the MT kernel
	{
            DEBUG_CUDA_CALL(cuModuleGetFunction(multistepKernel, multistepMod, "_Z35CalcField_MTkernel_CurvatureComputeILj32EEvP7Vector2IfEPfP11pointChargeIfEjjjjf"));
            DEBUG_CUDA_CALL(cuModuleGetFunction(singlestepKernel, singlestepMod, "_Z35CalcField_MTkernel_CurvatureComputeILj1EEvP7Vector2IfEPfP11pointChargeIfEjjjjf"));
	}
	else if(useMT)	// Has the wrapper padded the memory for the MT kernel?
	{
            DEBUG_CUDA_CALL(cuModuleGetFunction(multistepKernel, multistepMod, "_Z18CalcField_MTkernelILj32EEvP7Vector2IfEPfP11pointChargeIfEjjjjf"));
            DEBUG_CUDA_CALL(cuModuleGetFunction(singlestepKernel, singlestepMod, "_Z18CalcField_MTkernelILj1EEvP7Vector2IfEPfP11pointChargeIfEjjjjf"));
	}
	else	// Nope, just for the regular kernel
	{
            DEBUG_CUDA_CALL(cuModuleGetFunction(multistepKernel, multistepMod, "CalcField_SPkernel"));
            DEBUG_CUDA_CALL(cuModuleGetFunction(singlestepKernel, singlestepMod, "CalcField_SPkernel"));
	}
    return CUDA_SUCCESS;
}

template<>
CUresult CalcField_selectKernel<double>(CUmodule multistepMod, CUmodule singlestepMod,
                        CUfunction *multistepKernel, CUfunction *singlestepKernel,
                        bool useMT, bool useCurvature)
{
    if(useCurvature)
	{
        // Temporarily, the double-prcision, curvature kernel is not available
        return CUDA_ERROR_NOT_FOUND;
	}
	else if(useMT)	// Has the wrapper padded the memory for the MT kernel?
	{
            cuModuleGetFunction(multistepKernel, multistepMod, "CalcField_MTkernel_DP");
            cuModuleGetFunction(singlestepKernel, singlestepMod, "CalcField_MTkernel_DP");
	}
	else	// Nope, just for the regular kernel
	{
            cuModuleGetFunction(multistepKernel, multistepMod, "CalcField_DPkernel");
            cuModuleGetFunction(singlestepKernel, singlestepMod, "CalcField_DPkernel");
	}
    return CUDA_SUCCESS;
}

CUresult CalcField_loadModules(CUmodule *multistepMod, CUmodule *singlestepMod)
{
	// Attempt to first load a cubin module. If that fails, load the slower ptx module
	CUresult errCode;
	if(cuModuleLoad(singlestepMod, "Electrostatics.cubin") != CUDA_SUCCESS)
	{
		// Try to load from ptx code
		errCode = cuModuleLoad(singlestepMod, "Electrostatics.ptx");
		if(errCode != CUDA_SUCCESS)
			return errCode;
	}
	if(cuModuleLoad(multistepMod, "Electrostatics_Multistep.cubin") != CUDA_SUCCESS)
	{
		// Try to load from ptx code
		errCode = cuModuleLoad(multistepMod, "Electrostatics.ptx");
		if(errCode != CUDA_SUCCESS)
			return errCode;
	}

   
    return CUDA_SUCCESS;
}


// Calls the kernel, given pointers to device memory
// For this function to execute correctly, device memory must already be allocated and relevant data copied to device
// The non-wrapped function is useful when recalculating lines with memory already allocated

#include "X-Compat/HPC timing.h"
namespace CalcFieldEs
{

template<class T>
 CUresult Core(Vec3SOA<CUdeviceptr> &fieldLines, unsigned int steps, unsigned int nLines,
						unsigned int xyPitch, unsigned int zPitch,
						CUdeviceptr pointCharges, unsigned int points,
						T resolution, bool useMT, bool useCurvature,
						GPUkernels kernels)
{
	unsigned int bX = useMT ? BLOCK_X_MT : BLOCK_X;
	unsigned int bY = useMT ? BLOCK_Y_MT : 1;
	// Compute dimension requirements
	dim3 block(bX, bY, 1),
		grid( ((unsigned int)nLines + bX - 1)/bX, 1, 1 );
	CUresult errCode = CUDA_SUCCESS;

	// Load the multistep kernel parameters
	// Although device pointers are passed as CUdeviceptr, the kernel treats them as regular pointers.
	// Because of this, on 64-bit platforms, CUdevice ptr and regular pointers will have different sizes,
	// causing kernel parameters to be misaligned. For this reason, device pointers must be converted to
	// host pointers, and passed to the kernel accordingly.

	int offset = 0; unsigned int size = 0;
	Vector2<T> * xyParam = (Vector2<T> *)(size_t)fieldLines.xyInterleaved;
	size = sizeof(xyParam);
	DEBUG_CUDA_CALL(cuParamSetv(kernels.multistepKernel, offset, (void*)&xyParam, size));
	DEBUG_CUDA_CALL(cuParamSetv(kernels.singlestepKernel, offset, (void*)&xyParam, size));
	offset += size;

	T* zParam = (T*)(size_t) fieldLines.z;
	size = sizeof(zParam);
	DEBUG_CUDA_CALL(cuParamSetv(kernels.multistepKernel, offset, (void*)&zParam, size));
	DEBUG_CUDA_CALL(cuParamSetv(kernels.singlestepKernel, offset, (void*)&zParam, size));
	offset += size;

	pointCharge<T> * pointChargeParam = (pointCharge<T> *)(size_t)pointCharges;
	size = sizeof(pointChargeParam);
	DEBUG_CUDA_CALL(cuParamSetv(kernels.multistepKernel, offset, (void*)&pointChargeParam, size));
	DEBUG_CUDA_CALL(cuParamSetv(kernels.singlestepKernel, offset, (void*)&pointChargeParam, size));
	offset += size;

	size = sizeof(xyPitch);
	DEBUG_CUDA_CALL(cuParamSetv(kernels.multistepKernel, offset, (void*)&xyPitch, size));
	DEBUG_CUDA_CALL(cuParamSetv(kernels.singlestepKernel, offset, (void*)&xyPitch, size));
	offset += size;
	
	size = sizeof(zPitch);
	DEBUG_CUDA_CALL(cuParamSetv(kernels.multistepKernel, offset, (void*)&zPitch, size));
	DEBUG_CUDA_CALL(cuParamSetv(kernels.singlestepKernel, offset, (void*)&zPitch, size));
	offset += size;
	
	size = sizeof(points);
	DEBUG_CUDA_CALL(cuParamSetv(kernels.multistepKernel, offset, (void*)&points, size));
	DEBUG_CUDA_CALL(cuParamSetv(kernels.singlestepKernel, offset, (void*)&points, size));
	offset += size;

	const int fieldIndexParamSize = sizeof(unsigned int);
	// Do not set field index here, but before Launching the kernel
	const unsigned int fieldIndexParamOffset = offset;
	offset += fieldIndexParamSize;
	
	size = sizeof(resolution);
	DEBUG_CUDA_CALL(cuParamSetv(kernels.multistepKernel, offset, (void*)&resolution, size));
	DEBUG_CUDA_CALL(cuParamSetv(kernels.singlestepKernel, offset, (void*)&resolution, size));
	offset += size;

	DEBUG_CUDA_CALL(cuParamSetSize(kernels.multistepKernel, offset));
	DEBUG_CUDA_CALL(cuParamSetSize(kernels.singlestepKernel, offset));

	// Set Block Dimensions
	DEBUG_CUDA_CALL(cuFuncSetBlockShape(kernels.multistepKernel, block.x, block.y, block.z));
	DEBUG_CUDA_CALL(cuFuncSetBlockShape(kernels.singlestepKernel, block.x, block.y, block.z));

	// LAUNCH THE KERNEL
	unsigned int i = 1;
	while(i < (steps - KERNEL_STEPS) )
	{
		DEBUG_CUDA_KERNEL_SYNC(cuCtxSynchronize());// <- Remove this to crash the video driver
		cuParamSetv(kernels.multistepKernel, fieldIndexParamOffset, (void*)&i, fieldIndexParamSize);
		cuLaunchGrid(kernels.multistepKernel, grid.x, grid.y);
		i += KERNEL_STEPS;
	}
	while(i < steps)
	{
		DEBUG_CUDA_KERNEL_SYNC(cuCtxSynchronize());// <- Remove this to crash the video driver
		cuParamSetv(kernels.singlestepKernel, fieldIndexParamOffset, (void*)&i, fieldIndexParamSize);
		cuLaunchGrid(kernels.singlestepKernel, grid.x, grid.y);
		i++;
	}
	DEBUG_CUDA_CALL(cuCtxSynchronize());

	return CUDA_SUCCESS;
 }

}//namespace CalcField

