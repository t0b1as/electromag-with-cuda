/*
 * Copyright (C) 2010 - Alexandru Gagniuc - <mr.nuke.me@gmail.com>
 * This file is part of ElectroMag.
 *
 * ElectroMag is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * ElectroMag is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 *  along with ElectroMag.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "stdafx.h"
#if !defined(__CYGWIN__) // Stupid, I know, but it's a fact of life
#include <omp.h>
#endif
#include "./../../GPGPU_Segment/src/CL_Manager.hpp"
#include "Electromag utils.h"
#include "Graphics_dynlink.h"
#include <SOA_utils.hpp>
#include <thread>

//using namespace std;
// Use float or double; 16-bit single will generate errors
#define FPprecision float

struct SimulationParams
{
    size_t nx;          // Number of lines on the x direction
    size_t ny;          // Number of lines on the y direction
    size_t nz;          // Number of lines on the z direction
    size_t pStatic;     // Number of static point charges
    size_t pDynamic;    // Number of dynamic charge elements
    size_t len;         // Number of steps of a field line
};
// Default size for comparison with CPU performance
SimulationParams DefaultParams = {128, 128, 1, 1024, 0, 2500};
// Expect to fail on systems with under 3GB
SimulationParams EnhancedParams = {256, 112, 1, 2048, 0, 5000};
// Expect to fail on systems with under 6GB
SimulationParams ExtremeParams = {256, 256, 1, 2048, 0, 5000};
// Requires minimum 16GB system RAM + host buffers
SimulationParams InsaneParams = {512, 512, 1, 2048, 0, 5000};
// Requires minimum 24GB system RAM + host buffers
SimulationParams FuckingInsaneParams = {1024, 1024, 1, 5120, 0, 10000};
// Should work acceptably on most multi-core CPUs
SimulationParams CpuModeParams = {64, 64, 1, 1000, 0, 1000};
SimulationParams MicroParams = {16, 16, 1, 1000, 0, 1000};
SimulationParams BogoParams = {16, 16, 1, 50, 0, 500};
enum ParamLevel {__bogo, __micro, __cpu, __normal, __enhanced, __extreme,
                 __insane, __fuckingInsane
                };


void TestCL(Vector3<Array<float> >& fieldLines,
            Array<electro::pointCharge<float> >& pointCharges,
            size_t n, float resolution,  perfPacket& perfData,
            bool useCurvature);

using std::endl;
using std::cout;
using std::cerr;
using std::this_thread::sleep_for;
using std::chrono::milliseconds;
// to redirect stdout and stderr to out.txt use:
//              >out.txt  2>&1
int main ( int argc, char* argv[] )
{
    cout<<" Electromagnetism simulation application"<<endl;
    cout<<" Compiled on "<<__DATE__<<" at "<<__TIME__<<endl;

    OpenCL::GlobalClManager.ListAllDevices();


#ifndef _DEBUG
    //freopen( "file.txt", "w", stderr );
#endif//DEBUG

    ParamLevel paramLevel = __normal;

    SimulationParams simConfig = DefaultParams;
    bool saveData = false, CPUenable = false, GPUenable = true, display = true;
    bool useCurvature = true;
    bool visualProgressBar = false;
    bool randseed = false;
    bool randfieldinit = false;
    bool debugData = false;
    bool regressData = false;
    // Precision to use
    bool useCpuDP = false;
    bool useGpgpuDP = false;
    // OpenCL devel tests?
    bool clMode = false;
    // Get command-line options;
    for ( int i = 1; i < argc; i++ )
    {
        if ( !strcmp ( argv[i], "--cpu" ) )
            CPUenable = true;
        else if ( !strcmp ( argv[i], "--gpudisable" ) )
        {
            GPUenable = false;
            CPUenable = true;
        }
        else if ( !strcmp ( argv[i], "--save" ) )
            saveData = true;
        else if ( !strcmp ( argv[i], "--nodisp" ) )
            display = false;
        else if ( !strcmp ( argv[i], "--bogo" ) )
        {
            if ( paramLevel == __normal ) paramLevel = __bogo;
        }
        else if ( !strcmp ( argv[i], "--micro" ) )
        {
            if ( paramLevel == __normal ) paramLevel = __micro;
        }
        else if ( !strcmp ( argv[i], "--enhanced" ) )
        {
            if ( paramLevel < __enhanced ) paramLevel = __enhanced;
        }
        else if ( !strcmp ( argv[i], "--extreme" ) )
        {
            if ( paramLevel < __extreme ) paramLevel = __extreme;
        }
        else if ( !strcmp ( argv[i], "--insane" ) )
        {
            if ( paramLevel < __insane ) paramLevel = __insane;
        }
        else if ( !strcmp ( argv[i], "--fuckingInsane" ) )
        {
            if ( paramLevel < __fuckingInsane ) paramLevel = __fuckingInsane;
        }
        else if ( !strcmp ( argv[i], "--GUI" ) )
            visualProgressBar = true;
        else if ( !strcmp ( argv[i], "--randseed" ) )
            randseed = true;
        else if ( !strcmp ( argv[i], "--randfieldinit" ) )
            randfieldinit = true;
        else if ( !strcmp ( argv[i], "--postrundebug" ) )
            debugData = true;
        else if ( !strcmp ( argv[i], "--autoregress" ) )
            regressData = true;
        else if ( !strcmp ( argv[i], "--cpuprecision=double" ) )
            useCpuDP = true;
        else if ( !strcmp ( argv[i], "--gpuprecision=double" ) )
            useGpgpuDP = true;
        else if ( !strcmp ( argv[i], "--clmode" ) )
            clMode = true;
        else
            cout<<" Ignoring unknown argument: "<<argv[i]<<endl;
    }

    Render::Renderer* FieldDisplay = 0;
    // Do we need to load the graphicsModule?
    if ( display )
    {
        Graphics::ModuleLoadCode errCode;
        errCode = Graphics::LoadModule();
        if ( errCode != Graphics::SUCCESS )
        {
            cerr<<" Could not load graphhics module. Rendering disabled"<<endl;
            display = false;
        }
        else
        {
            FieldDisplay = Graphics::CreateFieldRenderer();
        }
    }

    CPUID::CpuidString cpuString;
    CPUID::GetCpuidString ( &cpuString );

    CPUID::CpuidFeatures cpuInfo;
    CPUID::GetCpuidFeatures ( &cpuInfo );

    const char *support[2] = {"not supported", "supported"};

    //freopen("log.bs.txt", "w", stderr);
    std::clog<<" Processor:\t";
    std::clog.write ( cpuString.IDString, sizeof ( cpuString.IDString ) );
    std::clog<<endl;
    std::clog<<" SSE3:  \t"<<support[cpuInfo.SSE3]<<endl;
    std::clog<<" SSSE3: \t"<<support[cpuInfo.SSSE3]<<endl;
    std::clog<<" SSE4.1:\t"<<support[cpuInfo.SSE41]<<endl;
    std::clog<<" SSE4.2:\t"<<support[cpuInfo.SSE42]<<endl;
    std::clog<<" AVX256:\t"<<support[cpuInfo.AVX]<<endl;

    // Now that checks are performed, start the Frontend
    //if(visualProgressBar) MainGUI.StartAsync();

    GPUenable = false;
    CPUenable=true;
    // Statistics show that users are happier when the program outputs fun
    // information abot their toys

    // Set correct parameter configuration
    switch ( paramLevel )
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
    case __cpu: //Fall Through
    default:
        simConfig = CpuModeParams;
    }
    // Initialze data containers
    size_t nw = ( int ) simConfig.nx,
                nh = ( int ) simConfig.ny,
                     nd = ( int ) simConfig.nz,
                          n = nh * nw * nd,
                              p = ( int ) simConfig.pStatic,
                                  len = ( int ) simConfig.len;
    Vector3 <Array<FPprecision> > CPUlines, GPUlines;
    Array<electro::pointCharge<FPprecision> > charges ( p, 256 );
    // Only allocate memory if cpu comparison mode is specified
    if ( GPUenable ) GPUlines.AlignAlloc ( n*len );
    if ( CPUenable ) CPUlines.AlignAlloc ( n*len );
    perfPacket CPUperf = {0, 0}, GPUperf = {0, 0};
    std::ofstream data, regress;
    if ( saveData )
        data.open ( "data.txt" );
    //MainGUI.RegisterProgressIndicator((double * volatile)&CPUperf.progress);

    // Do not activate if memory allocation fails
    if ( !CPUlines.GetSize() ) CPUenable = false;
    if ( !GPUlines.GetSize() ) GPUenable = false;

    InitializePointChargeArray ( charges, p, randseed );

    // init starting points
    Vector3 <Array<FPprecision> > *arrMain;
    if ( GPUenable ) arrMain = &GPUlines;
    else if ( CPUenable ) arrMain = &CPUlines;
    else
    {
        cerr<<" Could not allocate sufficient memory. Halting execution."<<endl;
        size_t neededRAM = n*len*sizeof ( Vector3<FPprecision> ) /1024/1024;
        cerr<<" "<<neededRAM<<" MB needed for initial allocation"<<endl;
        return 666;
    }

    // Initialize the starting points
    InitializeFieldLineArray ( *arrMain, n, nw, nh, nd, randfieldinit );

    // If both CPU and GPU modes are selected, the GPU array will have been
    // initialized first
    // Copy the same starting values to the CPU array
    if ( CPUenable && GPUenable )
        CopyFieldLineArray ( CPUlines, GPUlines, 0, n );

    // Run calculations
    long long freq, start, end;
    double GPUtime = 0, CPUtime = 0;
    QueryHPCFrequency ( &freq );

    if (clMode && CPUenable)
    {
        //StartConsoleMonitoring ( &CPUperf.progress );
        TestCL ( CPUlines, charges, n, 1.0, CPUperf, useCurvature );
        CPUperf.progress = 1.0;
    }
    else
    {

        FPprecision resolution = 1;
        if ( GPUenable )
        {
            cout<<" GPU"<<endl;
        }
        if ( CPUenable )
        {
            StartConsoleMonitoring ( &CPUperf.progress );
            QueryHPCTimer ( &start );
            CalcField_CPU ( CPUlines, charges, n, resolution, CPUperf,
                            useCurvature );
            QueryHPCTimer ( &end );
            CPUperf.progress = 1;
            cout<<" CPU kernel execution time:\t"
                <<CPUperf.time<<" seconds"<<endl;
            cout<<" Effective performance:\t\t"
                <<CPUperf.performance<<" GFLOP/s"<<endl;
            CPUtime = double ( end-start ) /freq;
            cout<<" True kernel execution time:\t"<<CPUtime<<" seconds"<<endl;
            if ( GPUenable )
            {
                cout<<" Effective speedup:\t\t"
                    <<GPUperf.performance/CPUperf.performance<<" x"<<endl;
                cout<<" Realistic speedup:\t\t"
                    <<CPUtime/GPUtime<<" x"<<endl;
            }
        }
    }

    FieldRenderer::GLpacket GLdata;
    volatile bool * shouldIQuit = 0;
    if ( display )
    {
        cout<<"Initializing display"<<endl;
        GLdata.charges = ( Array<electro::pointCharge<float> >* ) &charges;
        GLdata.lines = ( Vector3<Array<float> >* ) arrMain;
        GLdata.nlines = n;
        GLdata.lineLen = len;
        GLdata.elementSize = sizeof ( FPprecision );
        // Before: FieldDisp.RenderPacket(GLdata);
        FieldRenderer::FieldRenderCommData GLmessage;
        GLmessage.messageType = FieldRenderer::SendingGLData;
        GLmessage.commData = ( void* ) &GLdata;
        FieldDisplay->SendMessage ( &GLmessage );

        // Before: FieldDisp.SetPerfGFLOP(GPUperf.performance);
        FieldRenderer::FieldRenderCommData PerfMessage;
        PerfMessage.messageType = FieldRenderer::SendingPerfPointer;
        PerfMessage.commData = ( void* ) GPUenable ? (&GPUperf.performance) :
                               (&CPUperf.performance);
        FieldDisplay->SendMessage ( &PerfMessage );

        // Get ready to quit flag
        FieldRenderer::FieldRenderCommData quitMessage;
        quitMessage.messageType = FieldRenderer::RequestQuitFlag;
        FieldDisplay->SendMessage ( &quitMessage );
        shouldIQuit = ( bool* ) quitMessage.commData;

        try
        {
            cout<<" Starting display"<<endl;
            FieldDisplay->StartAsync();
        }
        catch ( char * errString )
        {
            cerr<<" Could not initialize field rendering"<<endl;
            cerr<<errString<<endl;
        }
    }
    else
    {
        cout<<" Skipping display"<<endl;
    }

    // do stuff here
    // This will generate files non-worthy of FAT32 or non-RAID systems
    if ( saveData && ( CPUenable || GPUenable ) )
    {
        cout<<" Beginning save procedure"<<endl;
        for ( size_t line = 0; line < n; line++ )
        {
            for ( size_t step = 0; step < len; step++ )
            {
                int i = step*n + line;
                if ( CPUenable )
                {
                    data<<" CPUL ["<<line<<"]["<<step<<"] x: "<<CPUlines[i].x
                    <<" y: "<<CPUlines[i].y<<" z: "<<CPUlines[i].z
                    <<endl;
                }
                if ( GPUenable )
                {
                    data<<" GPUL ["<<line<<"]["<<step<<"] x: "<<GPUlines[i].x
                    <<" y: "<<GPUlines[i].y<<" z: "<<GPUlines[i].z
                    <<endl;
                }
            }
            float percent = ( float ) line/n*100;
            cout<<percent<<" %complete"<<endl;
        }
        cout<<" Save procedure complete"<<endl;
    }

    // Save points that are significanlty off for regression analysis
    if ( regressData && CPUenable && GPUenable )
    {
        regress.open ( "regression.txt" );//, ios::app);
        cout<<" Beginning verfication procedure"<<endl;
        for ( size_t line = 0; line < n; line++ )
        {
            // Looks for points that are close to the CPU value
            // but suddenly jumps off
            // This ususally exposes GPU kernel syncronization bugs
            size_t step = 0;
            do
            {
                size_t i = step*n + line;
                size_t iLast = ( step-1 ) *n + line;
                // Calculate the distance between the CPU and GPU point
                float offset3D = vec3Len ( vec3 ( CPUlines[i],GPUlines[i] ) );
                if ( offset3D > 0.1f )
                {
                    regress<<" CPUL ["<<line<<"]["<<step-1<<"] x: "
                    <<CPUlines[iLast].x<<" y: "<<CPUlines[iLast].y<<" z: "
                    <<CPUlines[iLast].z
                    <<endl
                    <<" GPUL ["<<line<<"]["<<step-1<<"] x: "
                    <<GPUlines[iLast].x<<" y: "<<GPUlines[iLast].y
                    <<" z: "<<GPUlines[iLast].z
                    <<endl
                    <<" 3D offset: "
                    <<vec3Len ( vec3 ( CPUlines[iLast],GPUlines[iLast] ) )
                    <<endl;
                    regress<<" CPUL ["<<line<<"]["<<step<<"] x: "
                    <<CPUlines[i].x<<" y: "<<CPUlines[i].y<<" z: "
                    <<CPUlines[i].z
                    <<endl
                    <<" GPUL ["<<line<<"]["<<step<<"] x: "<<GPUlines[i].x
                    <<" y: "<<GPUlines[i].y<<" z: "<<GPUlines[i].z
                    <<endl
                    <<" 3D offset: "<<offset3D
                    <<endl<<endl;
                    // If a leap is found; skip the current line
                    break;
                }
            }
            while ( ++step < len );
            // Small anti-boredom indicator
            if ( ! ( line%256 ) )
            {
                cout<<" "<< ( double ) line/n*100<<" % complete"<<endl;
            }
        }
        // When done, close file to prevent system crashes from resulting
        // in incomplete regressions
        regress.close();
        cout<<" Verification complete"<<endl;
    }

    while ( debugData )
    {
        size_t line, step;
        std::cin>> line>>step;
        std::cin.clear();
        std::cin.ignore ( 100, '\n' );
        size_t i = step*n + line;

        if ( CPUenable )
        {
            cout<<" CPUL ["<<line<<"]["<<step<<"] x: "<<CPUlines[i].x
                <<" y: "<<CPUlines[i].y<<" z: "<<CPUlines[i].z<<endl;
        }
        if ( GPUenable )
        {
            cout<<" GPUL ["<<line<<"]["<<step<<"] x: "<<GPUlines[i].x
                <<" y: "<<GPUlines[i].y<<" z: "<<GPUlines[i].z<<endl;
        }
        if ( CPUenable && GPUenable )
        {
            float offset3D = vec3Len ( vec3 ( CPUlines[i],GPUlines[i] ) );
            cout<<" 3D offset: "<<offset3D<<endl;
        }
    }


    // Wait for renderer to close program if active; otherwise quit directly
    if ( display )
    {
        while ( !*shouldIQuit )
        {
            sleep_for(milliseconds( 1000 ));
        };
        FieldDisplay->KillAsync();
    }
    // do a DEBUG wait before cleaning resources
#ifdef _DEBUG
    if ( !display ) system ( "pause" );
#endif
    // Tidyness will help in the future
    CPUlines.Free();
    GPUlines.Free();
    charges.Free();
    return 0;
}
