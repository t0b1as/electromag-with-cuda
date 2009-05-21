#include "CpuID.h"

#ifdef WIN32
#include<intrin.h>
#endif
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
