#include "Electrostatics kernel.cuh"
#include "Electrostatics MT kernel.cuh"
#include "Config.h"

void templatizer(void)
{
    /*
     * Does the same job as templatizer in Electrostatics.cu, ecxept that it
     * compiles the kernels that execute several stes in one call. Placing the
     * kernels in different ptx modules, allows complications from the name
     * mangling scheme to be avoided by loading the appropriate kernel from
     * each module.
     */
    CalcField_MTkernel_CurvatureCompute<KERNEL_STEPS><<<1, 1>>>(0, 0, 0, 0, 0, 0, 0, 0);
    CalcField_MTkernel<KERNEL_STEPS><<<1, 1>>>(0, 0, 0, 0, 0, 0, 0, 0);
    CalcField_SPkernel<KERNEL_STEPS><<<1, 1>>>(0, 0, 0, 0, 0, 0, 0, 0);
    CalcField_DPkernel<KERNEL_STEPS><<<1, 1>>>(0, 0, 0, 0, 0, 0, 0, 0);
    CalcField_MTkernel_DP<KERNEL_STEPS><<<1, 1>>>(0, 0, 0, 0, 0, 0, 0, 0);

}