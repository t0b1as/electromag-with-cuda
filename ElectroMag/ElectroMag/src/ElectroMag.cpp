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
#include <omp.h>
#include "./../../GPGPU_Segment/src/CUDA Manager.h"
#include "./../../GPGPU_Segment/src/CL Manager.h"

 using namespace std;
 // Use float or double; 16-bit single will generate erors
 #define FPprecision float
static int finished = 0;

struct SimulationParams
{
	size_t nx;			// Number of lines on the x direction
	size_t ny;			// Number of lines on the y direction
	size_t nz;			// Number of lines on the z direction
	size_t pStatic;		// Number of static point charges
	size_t pDynamic;	// Number of dynamic charge elements
	size_t len;			// Number of steps of a field line
};
SimulationParams DefaultParams = {128, 128, 1, 1000, 0, 2500};			// Default size for comparison with CPU performance
SimulationParams EnhancedParams = {256, 112, 1, 2000, 0, 5000};			// Expect to fail on systems with under 3GB
SimulationParams ExtremeParams = {256, 256, 1, 2000, 0, 5000};			// Expect to fail on systems with under 6GB
SimulationParams InsaneParams = {512, 512, 1, 2000, 0, 5000};			// Requires minimum 16GB system RAM + host buffers
SimulationParams FuckingInsaneParams = {1024, 1024, 1, 5000, 0, 10000};	// Requires minimum 24GB system RAM + host buffers
SimulationParams CpuModeParams = {64, 64, 1, 1000, 0, 1000};			// Should work acceptably on most multi-core CPUs
//SimulationParams CpuModeParams = {16, 16, 1, 1000, 0, 1000};

// to redirect stdout and stderr to out.txt use:
//				>out.txt  2>&1
int main(int argc, char* argv[])
{
	cout<<" Electromagnetism simulation application"<<endl;
	cout<<" Compiled on "<<__DATE__<<" at "<<__TIME__<<endl;

	OpenCL::GlobalClManager.ListAllDevices();

#ifndef _DEBUG
	//freopen( "file.txt", "w", stderr );
#endif//DEBUG

	enum ParamLevel{__cpu, __normal, __enhanced, __extreme,  __insane, __fuckingInsane};
	ParamLevel paramLevel = __normal;

	SimulationParams simConfig = DefaultParams;
	bool saveData = false, CPUenable = false, GPUenable = true, display = true;
	bool useCurvature = true;
	bool visualProgressBar = false;
	bool randseed = false;
	bool randfieldinit = false;
	bool debugData = false;
	// Get command-line options;
	for(int i = 1; i < argc; i++)
	{
		if( !strcmp(argv[i], "cpu") )
			CPUenable = true;
		else if( !strcmp(argv[i], "gpudisable") )
			GPUenable = false;
		else if( !strcmp(argv[i], "save") )
			saveData = true;
		else if( !strcmp(argv[i], "nodisp") )
			display = false;
		else if( !strcmp(argv[i], "enhanced") )
			{if(paramLevel < __enhanced) paramLevel = __enhanced;}
		else if( !strcmp(argv[i], "extreme") )
			{if(paramLevel < __enhanced) paramLevel = __extreme;}
		else if( !strcmp(argv[i], "insane") )
			{if(paramLevel < __insane) paramLevel = __insane;}
		else if( !strcmp(argv[i], "fuckingInsane") )
			{if(paramLevel < __fuckingInsane) paramLevel = __fuckingInsane;}
		else if( !strcmp(argv[i], "GUI") )
			visualProgressBar = true;
		else if( !strcmp(argv[i], "randseed") )
			randseed = true;
		else if( !strcmp(argv[i], "randfieldinit") )
			randfieldinit = true;
		else if( !strcmp(argv[i], "postrundebug") )
			debugData = true;
		else
			cout<<" Ignoring unknown argument: "<<argv[i]<<endl;
	}

	
	CPUID::CpuidString cpuString;
	CPUID::GetCpuidString(&cpuString);
	
	CPUID::CpuidFeatures cpuInfo;
	CPUID::GetCpuidFeatures(&cpuInfo);

	char *support[2] = {"not supported", "supported"};

	//freopen("log.bs.txt", "w", stderr);
	clog<<" Processor:\t";
	clog.write(cpuString.IDString, sizeof(cpuString.IDString));
	clog<<endl;
	clog<<" SSE3:  \t"<<support[cpuInfo.SSE3]<<endl;
	clog<<" SSSE3: \t"<<support[cpuInfo.SSSE3]<<endl;
	clog<<" SSE4.1:\t"<<support[cpuInfo.SSE41]<<endl;
	clog<<" SSE4.2:\t"<<support[cpuInfo.SSE42]<<endl;
	clog<<" AVX256:\t"<<support[cpuInfo.AVX256]<<endl;
    
    // Now that checks are performed, start the Frontend
    if(visualProgressBar) MainGUI.StartAsync();

	// Statistics show that users are happier when the program outputs fun information abot their toys
	cuda::GlobalCudaManager.ListAllDevices();
	// Initialize GPUs
	int cudaDev = cuda::GlobalCudaManager.GetCompatibleDevNo();
	if(cudaDev > 0)
	{
		cout<<endl<<" Found "<<cudaDev<<" compatible devices."<<endl;
	}
	// disable GPU mode if no compatible device deteced
	if(!cudaDev)
	{
		GPUenable = false; CPUenable = true;	// And force CPU mode
		cout<<" Warning! No compatible GPU found. Using optimized CPU mode with reduced parameter set."<<endl;
		paramLevel = __cpu;
	}
	// Set correct parameter configuration
	switch(paramLevel)
	{
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
	int nw = (int)simConfig.nx, nh = (int)simConfig.ny, nd = (int)simConfig.nz,  n = nh * nw * nd, p = (int)simConfig.pStatic, len = (int)simConfig.len;
	Array<Vector3<FPprecision> > CPUlines, GPUlines;
	Array<pointCharge<FPprecision> > charges(p, 256);
	// Only allocate memory if cpu comparison mode is specified
	if(GPUenable) GPUlines.AlignAlloc(n*len);
	if(CPUenable) CPUlines.AlignAlloc(n*len);
	perfPacket CPUperf = {0, 0}, GPUperf = {0, 0};
	ofstream data, regress;
	if (saveData)
		data.open("data.txt");
    MainGUI.RegisterProgressIndicator((double * volatile)&CPUperf.progress);

	// Do not activate if memory allocation fails
	if(!CPUlines.GetSize()) CPUenable = false;
	if(!GPUlines.GetSize()) GPUenable = false;

	__int64 pseudoSeed; QueryHPCTimer(&pseudoSeed);
	if (randseed) srand(pseudoSeed%RAND_MAX);
        else srand(1);
	// Initialize values
	for(size_t i = 0; i < p ; i++)
	{
		charges[i].position.x = (FPprecision)(rand()-RAND_MAX/2)/RAND_MAX*10000;//(FPprecision)i + 1;
		charges[i].position.y = (FPprecision)(rand()-RAND_MAX/2)/RAND_MAX*10000;//(FPprecision)i + 1;
		charges[i].position.z = (FPprecision)(rand()-RAND_MAX/2)/RAND_MAX*10000;//(FPprecision)i + 1;
		charges[i].magnitude = (FPprecision)(rand()-RAND_MAX/10)/RAND_MAX; //0.001f;
	}
	// init starting points
	Array<Vector3<FPprecision> > *arrMain;
	if(GPUenable) arrMain = &GPUlines;
	else if(CPUenable) arrMain = &CPUlines;
	else 
	{
		cerr<<" Could not allocate sufficient memory. Halting execution."<<endl;
		size_t neededRAM = n*len*sizeof(Vector3<FPprecision>)/1024/1024;
		std::cerr<<" "<<neededRAM<<" MB needed for initial allocation"<<std::endl;
		return 666;
	}

    // Initialize field line grid
	if(randfieldinit)
	{
		// Random Filed line initialization
		for(size_t i = 0; i < n ; i++)
		{
			(*arrMain)[i].x = (FPprecision)(rand()-RAND_MAX/2)/RAND_MAX*10000;
			(*arrMain)[i].y = (FPprecision)(rand()-RAND_MAX/2)/RAND_MAX*10000;
			(*arrMain)[i].z = (FPprecision)(rand()-RAND_MAX/2)/RAND_MAX*10000;
		}
	}
	else
	{FPprecision zVal = (FPprecision)((FPprecision)-nd/2 + 1E-5);
	for(size_t k = 0; k < nd; k++, zVal++)// z coord
	{
		FPprecision yVal = (FPprecision)((FPprecision)-nh/2 + 1E-5);
		for(size_t j = 0; j < nh; j++, yVal++)// y coord
		{
			FPprecision xVal = (FPprecision)((FPprecision)-nw/2 + 1E-5);
			for(size_t i = 0; i < nw; i++, xVal++)// x coord
			{
				(*arrMain)[k*nw*nh + j*nw + i].x = (FPprecision) 10*xVal;
				(*arrMain)[k*nw*nh + j*nw + i].y = (FPprecision) 10*yVal;
				(*arrMain)[k*nw*nh + j*nw + i].z = (FPprecision) 10*zVal;
			}
		}
	}}
	

    // If both CPU and GPU modes are initialized, the GPU array will have been initialized
    // Copy the same starting values to the CPU array
	if(CPUenable && GPUenable)
	for(size_t i = 0; i < n; i++)
	{
		CPUlines[i] = GPUlines[i];
	}

	
	// Run calculations
	__int64 freq, start, end;
	double GPUtime, CPUtime;
	QueryHPCFrequency(&freq);

	FPprecision resolution = 1;
	if(GPUenable)
	{
		cout<<" GPU"<<endl;
		int failedFunctors;
		// If dynamic and nested OMP is not initialized, CalcField may only get
        // one OMP thread, severely hampering performance on multi-core/CPU systems
        omp_set_dynamic(true);
        omp_set_nested(true);
		#pragma omp parallel
		{
        #pragma omp sections nowait
        {
			// First section runs the calculations
            #pragma omp section
			{
				
				QueryHPCTimer(&start);
				failedFunctors = CalcField(GPUlines, charges, n, resolution, GPUperf, useCurvature);
				QueryHPCTimer(&end);
			}
			 // Second section monitors progress
            #pragma omp section
			if(!visualProgressBar)
            {
                const double step = (double)1/60;
                cout<<"[__________________________________________________________]"<<endl;
                for(double next=step; next < (1.0 - 1E-3); next += step)
                {
                    while(GPUperf.progress < next)
					{
                        Threads::Pause(500);
					}
                    cout<<".";
                    // Flush to make sure progress indicator is displayed immediately
                    cout.flush();
                }
				std::cout<<" Done"<<std::endl;
				cout.flush();
            }
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
		cout<<" CPU"<<endl;
        // If dynamic and nested OMP is not initialized, CalcField may only get
        // one OMP thread, severely hampering performance on multi-core/CPU systems
        omp_set_dynamic(true);
        omp_set_nested(true);
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
                cout<<"[__________________________________________________________]"<<endl;
                for(double next=step; next < (1.0 - 1E-3); next += step)
                {
                    while(CPUperf.progress < next)
                        Threads::Pause(500);
                    cout<<".";
                    // Flush to make sure progress indicator is displayed immediately
                    cout.flush();
                }
				std::cout<<" Done"<<std::endl;
				cout.flush();
            }
        
        }
		cout<<" CPU kernel execution time:\t"<<CPUperf.time<<" seconds"<<endl;
		cout<<" Effective performance:\t\t"<<CPUperf.performance<<" GFLOP/s"<<endl;
		CPUtime = double(end-start)/freq;
		cout<<" True kernel execution time:\t"<<CPUtime<<" seconds"<<endl;
		if(GPUenable)
		{
			cout<<" Effective speedup:\t\t"<<GPUperf.performance/CPUperf.performance<<" x"<<endl;
			cout<<" Realistic speedup:\t\t"<<CPUtime/GPUtime<<" x"<<endl;
		}
	}

	if(GPUenable)
	for(int i = 0; i < GPUperf.stepTimes.GetSize()/timingSize; i++)
	{
		double *base = GPUperf.stepTimes.GetDataPointer() + timingSize*i;
		const double accountedOverhead = base[resAlloc] + base[kernelLoad] + base[xyHtoH] + base[xyHtoD] + base[zHtoH] + base[zHtoD] +
			base[xyDtoH] + base[xyHtoHb] + base[zDtoH] + base[zHtoHb] + base[mFree];

		cout<<endl<<" Execution unit "<<i<<endl;
																	
		cout<<" ==== Operation ======= Batch size ==== Time ========== Speed======="<<endl;
		cout<<" xy Device to host\t"<<base[xySize]/1024/1024<<"\t\t"<<base[xyDtoH] <<"\t"<<base[xySize]/1024/1024/base[xyDtoH]<<endl;
		cout<<" z  Device to host\t"<<base[zSize] /1024/1024<<"\t\t"<<base[zDtoH]  <<"\t"<<base[zSize] /1024/1024/base[zDtoH] <<endl;
		cout<<" xy Host to host  \t"  <<base[xySize]/1024/1024<<"\t\t"<<base[xyHtoHb]<<"\t"<<base[xySize]/1024/1024/base[xyHtoHb]<<endl;
		cout<<" z  Host to host  \t"  <<base[zSize] /1024/1024<<"\t\t"<<base[zHtoHb] <<"\t"<<base[zSize] /1024/1024/base[zHtoHb]<<endl;
		cout<<" ==================================================================="<<endl;
		cout<<" kernel execution    "<<base[kernelExec]<<"s"<<endl;
		cout<<" kernelLoad overhead "<<base[kernelLoad]<<"s"<<endl;
		cout<<" resAlloc   overhead "<<base[resAlloc]<<"s"<<endl;
		cout<<" Associated overhead "<<accountedOverhead<<"s"<<endl;
		cout<<" Unacounted overhead "<<GPUtime - accountedOverhead - base[kernelExec]<<"s"<<endl;
			
	}

	GLpacket<FPprecision> GLdata;
	GLdata.charges = &charges;
	GLdata.lines = arrMain;
	GLdata.nlines = n;
	GLdata.lineLen = len;
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
		cout<<" Beginning save procedure"<<endl;
		for(int line = 0; line < n; line++)
		{
			for(int step = 0; step < len; step++)
			{
				int i = step*n + line;
				if(CPUenable)data<<" CPUL ["<<line<<"]["<<step<<"] x: "<<CPUlines[i].x<<" y: "<<CPUlines[i].y<<" z: "<<CPUlines[i].z<<endl;
				if(GPUenable)data<<" GPUL ["<<line<<"]["<<step<<"] x: "<<GPUlines[i].x<<" y: "<<GPUlines[i].y<<" z: "<<GPUlines[i].z<<endl;
			}
			float percent = (float)line/n*100;
			cout<<percent<<" %complete"<<endl;
		}
		cout<<" Save procedure complete"<<endl;
	}
	
    // Save points that are significanlty off for regression analysis
	if(CPUenable && GPUenable)
	{
		regress.open("regression.txt");//, ios::app);
		cout<<" Beginning verfication procedure"<<endl;
		for(int line = 0; line < n; line++)
		{
            // Looks for points that are close to the GPU value, but suddenly jump
            // off; This ususally exposes GPU kernel syncronization bugs
			int step = 0;
			do
			{
				int i = step*n + line;
				int iLast = (step-1)*n + line;
                // Calculate the distance between the CPU and GPU point
				float offset3D = vec3Len(vec3(CPUlines[i],GPUlines[i]));
				if( offset3D > 0.1f)
				{
					regress<<" CPUL ["<<line<<"]["<<step-1<<"] x: "<<CPUlines[iLast].x<<" y: "<<CPUlines[iLast].y<<" z: "<<CPUlines[iLast].z<<endl\
						<<" GPUL ["<<line<<"]["<<step-1<<"] x: "<<GPUlines[iLast].x<<" y: "<<GPUlines[iLast].y<<" z: "<<GPUlines[iLast].z<<endl\
						<<" 3D offset: "<<vec3Len(vec3(CPUlines[iLast],GPUlines[iLast]))<<endl;
					regress<<" CPUL ["<<line<<"]["<<step<<"] x: "<<CPUlines[i].x<<" y: "<<CPUlines[i].y<<" z: "<<CPUlines[i].z<<endl\
						<<" GPUL ["<<line<<"]["<<step<<"] x: "<<GPUlines[i].x<<" y: "<<GPUlines[i].y<<" z: "<<GPUlines[i].z<<endl\
						<<" 3D offset: "<<offset3D<<endl<<endl;
                    // If a leap is found; skip the current line
					break;
				}
			}while(++step < len);
            // Small anti-boredom indicator
			if(!(line%256)) cout<<" "<<(double)line/n*100<<" % complete"<<endl;
		}
        // When done, close file to prevent system crashes from resulting in incomplete regressions
		regress.close();
		cout<<" Verification complete"<<endl;
	}

	while(debugData)
	{
		size_t line, step;
		cin>> line>>step;
		cin.clear();
		cin.ignore(100, '\n');
		size_t i = step*n + line;
		
		if(CPUenable) cout<<" CPUL ["<<line<<"]["<<step<<"] x: "<<CPUlines[i].x<<" y: "<<CPUlines[i].y<<" z: "<<CPUlines[i].z<<endl;
		if(GPUenable) cout<<" GPUL ["<<line<<"]["<<step<<"] x: "<<GPUlines[i].x<<" y: "<<GPUlines[i].y<<" z: "<<GPUlines[i].z<<endl;
		if(CPUenable && GPUenable)
		{
			float offset3D = vec3Len(vec3(CPUlines[i],GPUlines[i]));
			cout<<" 3D offset: "<<offset3D<<endl;
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
