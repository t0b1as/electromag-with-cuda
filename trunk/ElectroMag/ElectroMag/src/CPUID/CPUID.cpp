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
#include "CpuID.h"

#ifdef WIN32
#include<intrin.h>
#endif

namespace CPUID
{

#ifdef __linux__
#define __cpuid(out, infoType)\
	asm("cpuid": "=a" (out[0]), "=b" (out[1]), "=c" (out[2]), "=d" (out[3]): "a" (infoType));
#define __cpuidex(out, infoType, ecx)\
	asm("cpuid": "=a" (out[0]), "=b" (out[1]), "=c" (out[2]), "=d" (out[3]): "a" (infoType), "c" (ecx));
#endif

void GetCpuidString(CpuidString *stringStruct)
{
	int info[4];
	__cpuid(info, String);
	stringStruct->CPUInfo[0] = info[0];
	stringStruct->CPUInfo[1] = info[1];
	stringStruct->CPUInfo[2] = info[3];// 2->3 not an error
	stringStruct->CPUInfo[3] = info[2];// 3->2 not an error
};

void GetCpuidFeatures(CpuidFeatures *featureStruct)
{
	__cpuid(featureStruct->CPUInfo, FeatureSupport);
};

}//namespace CPUID