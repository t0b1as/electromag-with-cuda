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
#include "Config.h"


void templatizer(void)
{
    /*
     * The nvcc compiler will not compile kernels that are written as function
     * temlates. Therefore, we need to call each kernel template with the template
     * values that will be used in order for those kernels to be included in the
     * ptx code.
     * Although the call is done using the runtime API, wile the rest of the
     * application utilizes the driver API, this function will never get compiled
     * into an object file, and will never be linked into the application,
     * therefore no conflict between the two APIs should appear
     */
    CalcField_MTkernel_CurvatureCompute<1><<<1, 1>>>(0, 0, 0, 0, 0, 0, 0, 0);
    CalcField_MTkernel<1><<<1, 1>>>(0, 0, 0, 0, 0, 0, 0, 0);
    CalcField_SPkernel<1><<<1, 1>>>(0, 0, 0, 0, 0, 0, 0, 0);
    CalcField_MTkernel_DP<1><<<1, 1>>>(0, 0, 0, 0, 0, 0, 0, 0);
    CalcField_DPkernel<1><<<1, 1>>>(0, 0, 0, 0, 0, 0, 0, 0);

}