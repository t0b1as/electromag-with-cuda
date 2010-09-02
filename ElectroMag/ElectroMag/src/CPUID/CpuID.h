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


#ifndef _CPUID_H
#define _CPUID_H

namespace CPUID
{
enum CPUIDInfoType
{
	String = 0, FeatureSupport = 1,

};
struct CpuidString
{
	union
	{
		int CPUInfo[4];
		struct
		{
			int maxiType;
			char IDString[12];
		};
	};
};
struct CpuidFeatures
{
	union
	{
		int CPUInfo[4];
		struct
		{
			// First ID string
			unsigned SteppingID		:4;///< SteppingID
			unsigned Model			:4;///< Model
			unsigned Family			:4;///< Family
			unsigned TypeItl		:2;///< Processor Type
			unsigned Reserved11		:2;///< Reserved
			unsigned ExtendedModel	:4;///< Extended Model
			unsigned ExtendedFamily	:8;///< Extended Family
			unsigned Reserved12		:3;///< Reserved
			// Second ID string
			unsigned BrandIndex		:8;///< Brand Index
			unsigned QwordCFLUSH	:8;///< CLFLUSH cache line size / 8
			unsigned LogicProcCount	:8;///< Maximum number of addressable IDs for logical processors
			unsigned ApicID			:8;///< APIC Physical ID
			// Third ID string
			unsigned SSE3			:1;///< Streaming SIMD Extensions 3 support
			unsigned PCLMULDQ		:1;///< PCLMULDQ Instruction
            unsigned DTES64         :1;///< 64-bit Debug store
			unsigned MWAIT			:1;///< MONITOR/MWAIT
			unsigned CPLDebug		:1;///< CPL Qualified Debug Store
			unsigned VMX			:1;///< Virtual Machine Extensions
			unsigned SMX        	:1;///< Safer Mode Extensions
			unsigned EST            :1;///< Enhanced Intel Speed-Step Technology
			unsigned TM2        	:1;///< Thermal Monitor 2
			unsigned SSSE3          :1;///< Supplemental Streaming SIMD Extensions 3 support
			unsigned CNXTID 		:1;///< L1 Context ID
			unsigned Reserved31		:1;///< Reserved
			unsigned FMA256			:1;///<
			unsigned CMPXCHG16B		:1;///< CMPXCHG16B Instruction
			unsigned xTPR			:1;///< xTPR Update Control
			unsigned PDCM   		:1;///< Perform and Debug Capability (MSR)
			unsigned Reserved32		:2;///< Reserved
			unsigned DCA        	:1;///< Direct Cache Access
			unsigned SSE41			:1;///< Streaming SIMD Extensions 4.1 support
			unsigned SSE42			:1;///< Streaming SIMD Extensions 4.2 support
			unsigned x2APIC			:1;///< Extended xAPIC support
			unsigned MOVBE			:1;///< MOVBE Instruction
			unsigned POPCNT			:1;///< POPCNT Instruction
			unsigned Reserved33		:1;///< Reserved
			unsigned AES    		:1;///< AES instruction
			unsigned XSAVE			:1;///< XSAVE/XSTOR States
			unsigned OSXSAVE		:1;///< OS-enabled extended state management
			unsigned AVX			:1;///< AVX256 Extensions support
			unsigned Reserved34 	:3;///< Reserved
			// Fourth ID String
			unsigned FPU			:1;///< Floating-Point Unit on Chip
			unsigned VME			:1;///< Virtual-8086 Mode Enhancement
			unsigned DE				:1;///< Debugging Extensions
			unsigned PSE			:1;///< Page Size Extensions
			unsigned TSC			:1;///< Time Stamp Counter
			unsigned MSR			:1;///< RDMSR and WRMSR Support
			unsigned PAE			:1;///< Physical Address Extensions
			unsigned MCE			:1;///< Machine Check Exception
			unsigned Cx8			:1;///< CMPXCHG8B Inst.
			unsigned APIC			:1;///< On-Chip APIC Hardware
			unsigned Reserved41		:1;///< Reserved
			unsigned SEP			:1;///< Fast System Call (SYSENTER and SYSEXIT)
			unsigned MTTR			:1;///< Memory Type Range Registers
			unsigned PGE			:1;///< PTE Global Bit
			unsigned MCA			:1;///< Machine Check Architecture
			unsigned CMOV			:1;///< Conditional Move/Compare Instruction
			unsigned PAT			:1;///< Page Attribute Table
			unsigned PSE36			:1;///< 36-bit Page Size Extension
			unsigned PSN			:1;///< Processor Serial Number present and enabled
			unsigned CFLUSH			:1;///< CFLUSH Instruction
			unsigned Reserved42		:1;///< Reserved
			unsigned DS				:1;///< Debug Store
			unsigned ACPI			:1;///< Thermal Monitor and Clock Ctrl
			unsigned MMX			:1;///< MMX Technology
			unsigned FXSR			:1;///< FXSAVE/FXRSTOR
			unsigned SSE			:1;///< Streaming SIMD Extensions support
			unsigned SSE2			:1;///< Streaming SIMD Extensions 2 support
			unsigned SS				:1;///< Self Snoop
			unsigned HTT		 	:1;///< Hyper-threading technology
			unsigned TM			 	:1;///< Thermal Monitor
			unsigned Reserved43		:1;///< Reserved
			unsigned PBE		 	:1;///< Pending Break Enable
		};
	};
};

void GetCpuidString(CpuidString *stringStruct);
void GetCpuidFeatures(CpuidFeatures *featureStruct);

}//namespace CPUID

#endif//_CPUID_H
