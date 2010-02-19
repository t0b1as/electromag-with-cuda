/***********************************************************************************************
	Electromag - Electomagnestism simulation application using CUDA accelerated computing
	Copyright (C) 2009-2010 - Alexandru Gagniuc - <http:\\g-tech.homeserver.com\HPC.htm>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
	
	This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
***********************************************************************************************/
/**********************************************************************************************
	Uses freeglut 2.6.0 RC1; freeglut source available from: <http://freeglut.sourceforge.net/>
	freeglut may be licensed under different terms. Check freeglut.h for details
***********************************************************************************************/

#include "stdafx.h"
#include "Graphics/FieldRender.h"
#include "Graphics/FrontendGUI.h"
#if !defined(__CYGWIN__) // Stupid, I know, but it's a fact of life
#include <omp.h>
#endif
#include "./../../GPGPU_Segment/src/CUDA Manager.h"
#include "./../../GPGPU_Segment/src/CL Manager.h"
#include "Electromag utils.h"

 //using namespace std;
 // Use float or double; 16-bit single will generate erors
 #define FPprecision float

struct SimulationParams
{
	size_t nx;			// Number of lines on the x direction
	size_t ny;			// Number of lines on the y direction
	size_t nz;			// Number of lines on the z direction
	size_t pStatic;		// Number of static point charges
	size_t pDynamic;	// Number of dynamic charge elements
	size_t len;			// Number of steps of a field line
};
SimulationParams DefaultParams = {128, 128, 1, 1024, 0, 2500};			// Default size for comparison with CPU performance
SimulationParams EnhancedParams = {256, 112, 1, 2048, 0, 5000};			// Expect to fail on systems with under 3GB
SimulationParams ExtremeParams = {256, 256, 1, 2048, 0, 5000};			// Expect to fail on systems with under 6GB
SimulationParams InsaneParams = {512, 512, 1, 2048, 0, 5000};			// Requires minimum 16GB system RAM + host buffers
SimulationParams FuckingInsaneParams = {1024, 1024, 1, 5120, 0, 10000};	// Requires minimum 24GB system RAM + host buffers
SimulationParams CpuModeParams = {64, 64, 1, 1000, 0, 1000};			// Should work acceptably on most multi-core CPUs
SimulationParams MicroParams = {16, 16, 1, 1000, 0, 1000};
SimulationParams BogoParams = {16, 16, 1, 50, 0, 500};

// to redirect stdout and stderr to out.txt use:
//				>out.txt  2>&1
int main(int argc, char* argv[])
{
	std::cout<<" Electromagnetism simulation application"<<std::endl;
	std::cout<<" Compiled on "<<__DATE__<<" at "<<__TIME__<<std::endl;

	OpenCL::GlobalClManager.ListAllDevices();

#ifndef _DEBUG
	//freopen( "file.txt", "w", stderr );
#endif//DEBUG

	enum ParamLevel{__bogo, __micro, __cpu, __normal, __enhanced, __extreme,  __insane, __fuckingInsane};
	ParamLevel paramLevel = __bogo;

	SimulationParams simConfig = DefaultParams;
	bool saveData = false, CPUenable = false, GPUenable = true, display = true;
	bool useCurvature = true;
	bool visualProgressBar = false;
	bool randseed = false;
	bool randfieldinit = false;
	bool debugData = false;
	bool regressData = false;
	// Precision to use
	bool useCpuDP = false; bool useGpgpuDP = false;
	// Get command-line options;
	for(int i = 1; i < argc; i++)
	{
		if( !strcmp(argv[i], "--cpu") )
			CPUenable = true;
		else if( !strcmp(argv[i], "--gpudisable") )
            {GPUenable = false; CPUenable = true;}
		else if( !strcmp(argv[i], "--save") )
			saveData = true;
		else if( !strcmp(argv[i], "--nodisp") )
			display = false;
        else if( !strcmp(argv[i], "--bogo") )
			{if(paramLevel < __bogo) paramLevel = __bogo;}
        else if( !strcmp(argv[i], "--micro") )
			{if(paramLevel < __micro) paramLevel = __micro;}
		else if( !strcmp(argv[i], "--enhanced") )
			{if(paramLevel < __enhanced) paramLevel = __enhanced;}
		else if( !strcmp(argv[i], "--extreme") )
			{if(paramLevel < __enhanced) paramLevel = __extreme;}
		else if( !strcmp(argv[i], "--insane") )
			{if(paramLevel < __insane) paramLevel = __insane;}
		else if( !strcmp(argv[i], "--fuckingInsane") )
			{if(paramLevel < __fuckingInsane) paramLevel = __fuckingInsane;}
		else if( !strcmp(argv[i], "--GUI") )
			visualProgressBar = true;
		else if( !strcmp(argv[i], "--randseed") )
			randseed = true;
		else if( !strcmp(argv[i], "--randfieldinit") )
			randfieldinit = true;
		else if( !strcmp(argv[i], "--postrundebug") )
			debugData = true;
		else if( !strcmp(argv[i], "--autoregress") )
			regressData = true;
		else if( !strcmp(argv[i], "--cpuprecision=double") )
			useCpuDP = true;
		else if( !strcmp(argv[i], "--gpuprecision=double") )
			useGpgpuDP = true;
		else
			std::cout<<" Ignoring unknown argument: "<<argv[i]<<std::endl;
	}

	
	CPUID::CpuidString cpuString;
	CPUID::GetCpuidString(&cpuString);
	
	CPUID::CpuidFeatures cpuInfo;
	CPUID::GetCpuidFeatures(&cpuInfo);

	const char *support[2] = {"not supported", "supported"};

	//freopen("log.bs.txt", "w", stderr);
	std::clog<<" Processor:\t";
	std::clog.write(cpuString.IDString, sizeof(cpuString.IDString));
	std::clog<<std::endl;
	std::clog<<" SSE3:  \t"<<support[cpuInfo.SSE3]<<std::endl;
	std::clog<<" SSSE3: \t"<<support[cpuInfo.SSSE3]<<std::endl;
	std::clog<<" SSE4.1:\t"<<support[cpuInfo.SSE41]<<std::endl;
	std::clog<<" SSE4.2:\t"<<support[cpuInfo.SSE42]<<std::endl;
	std::clog<<" AVX256:\t"<<support[cpuInfo.AVX256]<<std::endl;
    
    // Now that checks are performed, start the Frontend
    if(visualProgressBar) MainGUI.StartAsync();

	// Statistics show that users are happier when the program outputs fun information abot their toys
	cuda::GlobalCudaManager.ListAllDevices();
	// Initialize GPUs
	int cudaDev = cuda::GlobalCudaManager.GetCompatibleDevNo();
	if(cudaDev > 0)
	{
		std::cout<<std::endl<<" Found "<<cudaDev<<" compatible devices."<<std::endl;
	}
	// disable GPU mode if no compatible device deteced
	if(!cudaDev)
	{
		GPUenable = false;
		std::cout<<" Warning! No compatible GPU found."<<std::endl;
		if(!CPUenable)
		{
			CPUenable = true; // And force CPU mode
			std::cout<<"Using optimized CPU mode with reduced parameter set."<<std::endl;
			paramLevel = __cpu;
		}
	}
	// Set correct parameter configuration
	switch(paramLevel)
	{
    case __bogo:
        simConfig = BogoParams;
        break;
    case __micro:
        simConfig = MicroParams;
        break;
	case __normal:
		simConfig = DefaultParams;
		break;
	case __enhanced:
		simConfig = EnhancedParams;
		break;
	case __extreme:
		simConfig = ExtremeParams;
		break;
	case __insane:
		simConfig = InsaneParams;
		break;
	case __fuckingInsane:
		simConfig = FuckingInsaneParams;
		break;
	case __cpu:	//Fall Through
	default:
		simConfig = CpuModeParams;
	}
	// Initialze data containers
	size_t nw = (int)simConfig.nx, nh = (int)simConfig.ny, nd = (int)simConfig.nz,  n = nh * nw * nd, p = (int)simConfig.pStatic, len = (int)simConfig.len;
    std::cout<<" nw"<<nw<<" nh "<<nh<<" nd "<<nd<<" ln "<<len<<" p "<<p<<std::endl;
	Array<Vector3<FPprecision> > CPUlines, GPUlines;
	Array<pointCharge<FPprecision> > charges(p, 256);
	// Only allocate memory if cpu comparison mode is specified
	if(GPUenable) GPUlines.AlignAlloc(n*len);
	if(CPUenable) CPUlines.AlignAlloc(n*len);
	perfPacket CPUperf = {0, 0}, GPUperf = {0, 0};
	std::ofstream data, regress;
	if (saveData)
		data.open("data.txt");
    MainGUI.RegisterProgressIndicator((double * volatile)&CPUperf.progress);

	// Do not activate if memory allocation fails
	if(!CPUlines.GetSize()) CPUenable = false;
	if(!GPUlines.GetSize()) GPUenable = false;

	InitializePointChargeArray(charges, p, randseed);

	// init starting points
	Array<Vector3<FPprecision> > *arrMain;
	if(GPUenable) arrMain = &GPUlines;
	else if(CPUenable) arrMain = &CPUlines;
	else 
	{
		std::cerr<<" Could not allocate sufficient memory. Halting execution."<<std::endl;
		size_t neededRAM = n*len*sizeof(Vector3<FPprecision>)/1024/1024;
		std::cerr<<" "<<neededRAM<<" MB needed for initial allocation"<<std::endl;
		return 666;
	}

	// Initialize the starting points
	InitializeFieldLineArray(*arrMain, n, nw, nh, nd, randfieldinit);

    // If both CPU and GPU modes are initialized, the GPU array will have been initialized
    // Copy the same starting values to the CPU array
	if(CPUenable && GPUenable) CopyFieldLineArray(CPUlines, GPUlines, 0, n);
	
	// Run calculations
	__int64 freq, start, end;
	double GPUtime = 0, CPUtime = 0;
	QueryHPCFrequency(&freq);

	FPprecision resolution = 1;
	if(GPUenable)
	{
		std::cout<<" GPU"<<std::endl;
		int failedFunctors;
		// If dynamic and nested OMP is not initialized, CalcField may only get
        // one OMP thread, severely hampering performance on multi-core/CPU systems
#		if !defined(__CYGWIN__)
        omp_set_dynamic(true);
        omp_set_nested(true);
#		endif
        #pragma omp parallel sections
        {
			// First section runs the calculations
            #pragma omp section
			{
				
				QueryHPCTimer(&start);
				failedFunctors = CalcField(GPUlines, charges, n, resolution, GPUperf, useCurvature);
				QueryHPCTimer(&end);
				// Make sure the next section terminates even if progress is not updated,
				// or is not updated entirely
				GPUperf.progress = 1;
			}
			 // Second section monitors progress
            #pragma omp section
			if(!visualProgressBar)
            {
                const double step = (double)1/60;
				std::cout<<"[__________________________________________________________]"<<std::endl;
                for(double next=step; next < (1.0 - 1E-3); next += step)
                {
                    while(GPUperf.progress < next)
					{
                        Threads::Pause(250);
					}
                    std::cout<<".";
                    // Flush to make sure progress indicator is displayed immediately
                    std::cout.flush();
                }
				std::cout<<" Done"<<std::endl;
				std::cout.flush();
            }
		}
		if(failedFunctors >= cudaDev) display = false;
		if(failedFunctors) std::cout<<" GPU Processing incomplete. "<<failedFunctors<<" functors out of "<<cudaDev<<" failed execution"<<std::endl;
		std::cout<<" GPU kernel execution time:\t"<<GPUperf.time<<" seconds"<<std::endl;
		std::cout<<" Effective performance:\t\t"<<GPUperf.performance<<" GFLOP/s"<<std::endl;
		GPUtime = double(end-start)/freq;
		std::cout<<" True kernel execution time:\t"<<GPUtime<<" seconds"<<std::endl;
	}
	if(CPUenable)
	{
		std::cout<<" CPU"<<std::endl;
        // If dynamic and nested OMP is not initialized, CalcField may only get
        // one OMP thread, severely hampering performance on multi-core/CPU systems
#		if !defined(__CYGWIN__)
        omp_set_dynamic(true);
        omp_set_nested(true);
#		endif
        #pragma omp parallel sections
        {
            // First section runs the calculations
            #pragma omp section
            {
                QueryHPCTimer(&start);
                CalcField_CPU(CPUlines, charges, n, resolution, CPUperf, useCurvature);
                QueryHPCTimer(&end);
            }
            // Second section monitors progress
            #pragma omp section
            if(!visualProgressBar)
            {
                const double step = (double)1/60;
                std::cout<<"[__________________________________________________________]"<<std::endl;
                for(double next=step; next < (1.0 - 1E-3); next += step)
                {
                    while(CPUperf.progress < next)
                        Threads::Pause(500);
                    std::cout<<".";
                    // Flush to make sure progress indicator is displayed immediately
                    std::cout.flush();
                }
				std::cout<<" Done"<<std::endl;
				std::cout.flush();
            }
        
        }
		std::cout<<" CPU kernel execution time:\t"<<CPUperf.time<<" seconds"<<std::endl;
		std::cout<<" Effective performance:\t\t"<<CPUperf.performance<<" GFLOP/s"<<std::endl;
		CPUtime = double(end-start)/freq;
		std::cout<<" True kernel execution time:\t"<<CPUtime<<" seconds"<<std::endl;
		if(GPUenable)
		{
			std::cout<<" Effective speedup:\t\t"<<GPUperf.performance/CPUperf.performance<<" x"<<std::endl;
			std::cout<<" Realistic speedup:\t\t"<<CPUtime/GPUtime<<" x"<<std::endl;
		}
	}

	if(GPUenable)
	for(size_t i = 0; i < GPUperf.stepTimes.GetSize()/timingSize; i++)
	{
		double *base = GPUperf.stepTimes.GetDataPointer() + timingSize*i;
		const double accountedOverhead = base[resAlloc] + base[kernelLoad] + base[xyHtoH] + base[xyHtoD] + base[zHtoH] + base[zHtoD] +
			base[xyDtoH] + base[xyHtoHb] + base[zDtoH] + base[zHtoHb] + base[mFree];

		std::cout<<std::endl<<" Execution unit "<<i<<std::endl;
																	
		std::cout<<" ==== Operation ======= Batch size ==== Time ========== Speed======="<<std::endl;
		std::cout<<" xy Device to host\t"<<base[xySize]/1024/1024<<"\t\t"<<base[xyDtoH] <<"\t"<<base[xySize]/1024/1024/base[xyDtoH]<<std::endl;
		std::cout<<" z  Device to host\t"<<base[zSize] /1024/1024<<"\t\t"<<base[zDtoH]  <<"\t"<<base[zSize] /1024/1024/base[zDtoH] <<std::endl;
		std::cout<<" xy Host to host  \t"  <<base[xySize]/1024/1024<<"\t\t"<<base[xyHtoHb]<<"\t"<<base[xySize]/1024/1024/base[xyHtoHb]<<std::endl;
		std::cout<<" z  Host to host  \t"  <<base[zSize] /1024/1024<<"\t\t"<<base[zHtoHb] <<"\t"<<base[zSize] /1024/1024/base[zHtoHb]<<std::endl;
		std::cout<<" ==================================================================="<<std::endl;
		std::cout<<" kernel execution    "<<base[kernelExec]<<"s"<<std::endl;
		std::cout<<" kernelLoad overhead "<<base[kernelLoad]<<"s"<<std::endl;
		std::cout<<" resAlloc   overhead "<<base[resAlloc]<<"s"<<std::endl;
		std::cout<<" Associated overhead "<<accountedOverhead<<"s"<<std::endl;
		std::cout<<" Unacounted overhead "<<GPUtime - accountedOverhead - base[kernelExec]<<"s"<<std::endl;
			
	}

	GLpacket GLdata;
	GLdata.charges = (Array<pointCharge<float> >*)&charges;
	GLdata.lines = (Array<Vector3<float> >*)arrMain;
	GLdata.nlines = n;
	GLdata.lineLen = len;
	GLdata.elementSize = sizeof(FPprecision);
	FieldDisp.RenderPacket(GLdata);
	FieldDisp.SetPerfGFLOP(GPUperf.performance);
    if(display)
	{
		try
		{
			FieldDisp.StartAsync();
		}
		catch(char * errString)
		{
			std::cerr<<" Could not initialize field rendering"<<std::endl;
			std::cerr<<errString<<std::endl;
		}
	}

	// do stuff here; This will generate files non-worthy of FAT32 or non-RAID systems
	if(saveData && (CPUenable || GPUenable))
	{
		std::cout<<" Beginning save procedure"<<std::endl;
		for(size_t line = 0; line < n; line++)
		{
			for(size_t step = 0; step < len; step++)
			{
				int i = step*n + line;
				if(CPUenable)data<<" CPUL ["<<line<<"]["<<step<<"] x: "<<CPUlines[i].x<<" y: "<<CPUlines[i].y<<" z: "<<CPUlines[i].z<<std::endl;
				if(GPUenable)data<<" GPUL ["<<line<<"]["<<step<<"] x: "<<GPUlines[i].x<<" y: "<<GPUlines[i].y<<" z: "<<GPUlines[i].z<<std::endl;
			}
			float percent = (float)line/n*100;
			std::cout<<percent<<" %complete"<<std::endl;
		}
		std::cout<<" Save procedure complete"<<std::endl;
	}
	
    // Save points that are significanlty off for regression analysis
	if(regressData && CPUenable && GPUenable)
	{
		regress.open("regression.txt");//, ios::app);
		std::cout<<" Beginning verfication procedure"<<std::endl;
		for(size_t line = 0; line < n; line++)
		{
            // Looks for points that are close to the CPU value, but suddenly jump
            // off; This ususally exposes GPU kernel syncronization bugs
			size_t step = 0;
			do
			{
				size_t i = step*n + line;
				size_t iLast = (step-1)*n + line;
                // Calculate the distance between the CPU and GPU point
				float offset3D = vec3Len(vec3(CPUlines[i],GPUlines[i]));
				if( offset3D > 0.1f)
				{
					regress<<" CPUL ["<<line<<"]["<<step-1<<"] x: "<<CPUlines[iLast].x<<" y: "<<CPUlines[iLast].y<<" z: "<<CPUlines[iLast].z<<std::endl\
						<<" GPUL ["<<line<<"]["<<step-1<<"] x: "<<GPUlines[iLast].x<<" y: "<<GPUlines[iLast].y<<" z: "<<GPUlines[iLast].z<<std::endl\
						<<" 3D offset: "<<vec3Len(vec3(CPUlines[iLast],GPUlines[iLast]))<<std::endl;
					regress<<" CPUL ["<<line<<"]["<<step<<"] x: "<<CPUlines[i].x<<" y: "<<CPUlines[i].y<<" z: "<<CPUlines[i].z<<std::endl\
						<<" GPUL ["<<line<<"]["<<step<<"] x: "<<GPUlines[i].x<<" y: "<<GPUlines[i].y<<" z: "<<GPUlines[i].z<<std::endl\
						<<" 3D offset: "<<offset3D<<std::endl<<std::endl;
                    // If a leap is found; skip the current line
					break;
				}
			}while(++step < len);
            // Small anti-boredom indicator
			if(!(line%256)) std::cout<<" "<<(double)line/n*100<<" % complete"<<std::endl;
		}
        // When done, close file to prevent system crashes from resulting in incomplete regressions
		regress.close();
		std::cout<<" Verification complete"<<std::endl;
	}

	while(debugData)
	{
		size_t line, step;
		std::cin>> line>>step;
		std::cin.clear();
		std::cin.ignore(100, '\n');
		size_t i = step*n + line;
		
		if(CPUenable) std::cout<<" CPUL ["<<line<<"]["<<step<<"] x: "<<CPUlines[i].x<<" y: "<<CPUlines[i].y<<" z: "<<CPUlines[i].z<<std::endl;
		if(GPUenable) std::cout<<" GPUL ["<<line<<"]["<<step<<"] x: "<<GPUlines[i].x<<" y: "<<GPUlines[i].y<<" z: "<<GPUlines[i].z<<std::endl;
		if(CPUenable && GPUenable)
		{
			float offset3D = vec3Len(vec3(CPUlines[i],GPUlines[i]));
			std::cout<<" 3D offset: "<<offset3D<<std::endl;
		}
	}


	// Wait for renderer to close program if active; otherwise quit directly
	if(display)
	{
		while(!shouldIQuit)
		{
			Threads::Pause(1000);
		};
		FieldDisp.KillAsync();
	}
	// do a DEBUG wait before cleaning resources, so resource usage can be evaluated
#ifdef _DEBUG
	if(!display) system("pause");
#endif
	// Tidyness will help in the future
	CPUlines.Free();
	GPUlines.Free();
	charges.Free();
	return 0;
}
