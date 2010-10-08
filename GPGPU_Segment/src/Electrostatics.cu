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
/*////////////////////////////////////////////////////////////////////////////////
 See Makefile for compilation options
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
     * Although the call is done using the runtime API, while the rest of the
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