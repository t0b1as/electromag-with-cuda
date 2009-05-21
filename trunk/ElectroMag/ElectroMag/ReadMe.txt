========================================================================
    CONSOLE APPLICATION : ElectroMag Project Overview
========================================================================

AppWizard has created this ElectroMag application for you.  

This file contains a summary of what you will find in each of the files that
make up your ElectroMag application.


ElectroMag.vcproj
    This is the main project file for VC++ projects generated using an Application Wizard. 
    It contains information about the version of Visual C++ that generated the file, and 
    information about the platforms, configurations, and project features selected with the
    Application Wizard.

ElectroMag.cpp
    This is the main application source file.
    
/////////////////////////////////////////////////////////////////////////////
Required compile setup:
	Visual Studio 2008
	CUDA toolkit 2.2 x64
	CUDA SDK 2.2 x64
	Intel C++ Compiler 11.0.72 with x86-64 extensions
	A very very very powerful computer (as of 2009)

Environment variables:

setx CUDA_SDK_INC_PATH "%NVSDKCUDA_ROOT%\common\inc"
setx CUDA_SDK_LIB_PATH "%NVSDKCUDA_ROOT%\common\lib"
setx CUDA_BIN_PATH_32 "%CUDA_BIN_PATH%\..\CUDA32\bin"
setx CUDA_LIB_PATH_32 "%CUDA_BIN_PATH%\..\CUDA32\lib"
/////////////////////////////////////////////////////////////////////////////
Other standard files:

StdAfx.h, StdAfx.cpp
    These files are used to build a precompiled header (PCH) file
    named ElectroMag.pch and a precompiled types file named StdAfx.obj.

/////////////////////////////////////////////////////////////////////////////
Update log:
Main program:
	- renamed "CPU Implement.cpp" to "CPU_Implement.cpp" for Linux cross-compatibility
CPU Implementation:
	- Introduced option te allocate memory aligned to 256-byte boundary for Array<T>
	- Created aligned memory allocation for main data arrays (performance more consistent)
	- Eliminated omp_set_num_threads() in CPU kernel (25% throughput increase)
	- Aligned main host arrays to 256-byte boundary
GPU Implementation
	- Fixed bug where multithreaded kernel would produce undeterministic errors due to syncronization error
	- Fixed issue with kernel splitting that could cause some data to not be processed.
	- Implemented kernel splitting, where a kernel will be executed multiple times fot successive sets
	of data when the original set cannot fully fit in global memory
	- Implemented and tested multi-GPU calculations
OpenGL Implementation:
	- Updated GL renderer to use a color arrays rather than glColor3f
	- Updated GL renderer to use VBOs if available (with GLEW)
	- Fixed problem where depth-buffering, blending, and antialiasing were not working

/////////////////////////////////////////////////////////////////////////////
Known issues
- The "non-multithreaded" kernel is about 7 to 8 % faster than the "multithreaded" kernel 
 when the data size is a multiple of the number of multiprocesors.
- Under 64-bit mode, when not compiling with -maxrregcount 18, 19, or 20, the "multithreaded"
 kernel uses 21 registers, which reduces its throughput significantly.
- With the 182.06 driver, non-paged to device memory copies seem to be capped around 3.3GB/s.
- Other unknown issues may exist under 32-bit mode, as I did not intensivley test
 under 32-bit mode.

Linux cross-compatibility issues:
- The code will need small adjustments to compile with GCC or Intel C++ under linux,
- Under Linux, enabling depth-buffering, blending, or antialiasing will cause a segmentation fault? - Issue may have been fixed, but not tested yet
- I am still not able to link the GPU code, so calls to funcions implemented in the
 "GPGPU Segment" must be commented out to link and run the program
	


/////////////////////////////////////////////////////////////////////////////
Other notes:

AppWizard uses "TODO:" comments to indicate parts of the source code you
should add to or customize.

/////////////////////////////////////////////////////////////////////////////
