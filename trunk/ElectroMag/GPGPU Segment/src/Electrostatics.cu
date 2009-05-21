/*////////////////////////////////////////////////////////////////////////////////
compile with:

//Debug
"$(CUDA_BIN_PATH)\nvcc.exe" -maxrregcount 16 --ptxas-options=-v -ccbin "$(VCInstallDir)bin" 
-c -D_DEBUG -DWIN64 -D_CONSOLE -D_MBCS -Xcompiler /EHsc,/W3,/nologo,/Od,/Zi,/RTC1,/MTd 
-I.\..\ElectroMag\src\ -I"$(CUDA_INC_PATH)" -I"$(CUDA_SDK_INC_PATH)" -I./ 
-o $(PlatformName)\$(ConfigurationName)\CUDA_Electrostatics.obj src\Electrostatics.cu


//Release:
"$(CUDA_BIN_PATH)\nvcc.exe" -maxrregcount 16 --ptxas-options=-v -ccbin "$(VCInstallDir)bin" 
-c -D_NDEBUG -DWIN64 -D_CONSOLE -D_MBCS -Xcompiler /EHsc,/W3,/nologo,/O2,/Zi,/MT -I.\..\ElectroMag\src\ 
-I"$(CUDA_INC_PATH)" -I"$(CUDA_SDK_INC_PATH)" -I./ 
-o $(PlatformName)\$(ConfigurationName)\CUDA_Electrostatics.obj src\Electrostatics.cu

$(NVIDIA_SDK_INC_PATH)

*/////////////////////////////////////////////////////////////////////////////////
#include "Electrostatics kernel.cuh"
#include "Electrostatics MT kernel.cuh"
// I'm not sure this is needed; causes problems under linux
//#include "CPGP Interop.h"


// Calls the kernel, given pointers to device memory
// For this function to execute correctly, device memory must already be allocated and relevant data copied to device
// The non-wrapped function is useful when recalculating lines with memory already allocated
template<class T>
 cudaError_t CalcField_core(Vec3SOA<T> &fieldLines, unsigned int steps, unsigned int lines,
						unsigned int xyPitchOffset, unsigned int zPitchOffset,
						pointCharge<T> * pointCharges, unsigned int points,
						T resolution, bool useMT, bool useCurvature)
{
	// The generic template does nothing.
	// There is no optimization for generic types
	return cudaErrorNotYetImplemented;
}
template<>
 cudaError_t CalcField_core<float>(Vec3SOA<float> &fieldLines, unsigned int steps, unsigned int lines,
					unsigned int xyPitchOffset, unsigned int zPitchOffset,
					pointCharge<float> * pointCharges, unsigned int points,
					float resolution, bool useMT, bool useCurvature)
{
	unsigned int bX = useMT ? BLOCK_X_MT : BLOCK_X;
	unsigned int bY = useMT ? BLOCK_Y_MT : 1;
	// Compute dimension requirements
	dim3 block(bX, bY, 1),
		grid( ((unsigned int)lines + bX - 1)/bX, 1, 1 );

	// Call the kernel
	if(useCurvature)
	{
		unsigned int i = 1;
		while(i < (steps - KERNEL_STEPS) )
		{
			CalcField_MTkernel_CurvatureCompute<KERNEL_STEPS><<<grid, block>>>(fieldLines.xyInterleaved, fieldLines.z,
				pointCharges, xyPitchOffset, zPitchOffset, points, i, resolution);
			i += KERNEL_STEPS;
		}
		while(i < steps)
		{
			CalcField_MTkernel_CurvatureCompute<1><<<grid, block>>>(fieldLines.xyInterleaved, fieldLines.z,
				pointCharges, xyPitchOffset, zPitchOffset, points, i, resolution);
			i++;
		}
	}
	else if(useMT)	// Has the wrapper padded the memory for the MT kernel?
	{
		unsigned int i = 1;
		while(i < (steps - KERNEL_STEPS) )
		{
			CalcField_MTkernel<KERNEL_STEPS><<<grid, block>>>(fieldLines.xyInterleaved, fieldLines.z,
				pointCharges, xyPitchOffset, zPitchOffset, points, i, resolution);
			i += KERNEL_STEPS;
		}
		while(i < steps)
		{
			CalcField_MTkernel<1><<<grid, block>>>(fieldLines.xyInterleaved, fieldLines.z,
				pointCharges, xyPitchOffset, zPitchOffset, points, i, resolution);
			i++;
		}
	}
	else	// Nope, just for the regular kernel
	{
		unsigned int i = 1;
		while(i < (steps - KERNEL_STEPS) )
		{
			CalcField_SPkernel<KERNEL_STEPS><<<grid, block>>>(fieldLines.xyInterleaved, fieldLines.z,
				pointCharges, xyPitchOffset, zPitchOffset, points, i, resolution);
			i += KERNEL_STEPS;
		}
		while(i < steps)
		{
			CalcField_SPkernel<1><<<grid, block>>>(fieldLines.xyInterleaved, fieldLines.z,
				pointCharges, xyPitchOffset, zPitchOffset, points, i, resolution);
			i++;
		}
	}

	return cudaSuccess;
}
template<>
 cudaError_t CalcField_core<double>(Vec3SOA<double> &fieldLines, unsigned int steps, unsigned int lines,
					unsigned int xyPitchOffset, unsigned int zPitchOffset,
					pointCharge<double> * pointCharges, unsigned int points,
					double resolution, bool useMT, bool useCurvature)
{
	// Compute dimension requirements
	unsigned int bX = useMT ? BLOCK_X_MT : BLOCK_X;
	unsigned int bY = useMT ? BLOCK_Y_MT : 1;
	// Compute dimension requirements
	dim3 block(bX, bY, 1),
		grid( ((unsigned int)lines + bX - 1)/bX, 1, 1 );

	if(useMT)
	{
		unsigned int i = 1;
		while(i < (steps - KERNEL_STEPS) )
		{
			CalcField_MTkernel_DP<KERNEL_STEPS><<<grid, block>>>(fieldLines.xyInterleaved, fieldLines.z,
				pointCharges, xyPitchOffset, zPitchOffset, points, i, resolution);
			i += KERNEL_STEPS;
		}
		while(i < steps)
		{
			CalcField_MTkernel_DP<1><<<grid, block>>>(fieldLines.xyInterleaved, fieldLines.z,
				pointCharges, xyPitchOffset, zPitchOffset, points, i, resolution);
			i++;
		}
	}
	else
	{
		// Call the kernel
		for(unsigned int i = 1; i < steps; i++)
		{
			CalcField_DPkernel<<<grid, block>>>(fieldLines.xyInterleaved, fieldLines.z,
				pointCharges, xyPitchOffset, zPitchOffset, points, i, resolution);
		}
	}
	return cudaSuccess;
 }


