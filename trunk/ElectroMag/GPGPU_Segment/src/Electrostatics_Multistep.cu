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