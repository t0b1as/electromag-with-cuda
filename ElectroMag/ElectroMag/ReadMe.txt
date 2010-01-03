========================================================================
    CONSOLE APPLICATION : ElectroMag Project Overview
========================================================================.
Copyright Alexandru Gagniuc

In order to compile,

1) In Config.mk
  ensure that CUDA_TOOLKIT_PATH and CUDA_SDK_PATH point
  to the locations of the CUDA toolkit and SDK, respectively.
2) in ElectroMag/Makefile-Intel.mk
  ensure that ICL_PATH points to Intel's C++ compiler bin directory
  NOTE: If not using Intel's C++ Compiler,
	in ElectroMag/Makefile.mk comment the folowing line
	include Makefile-Intel.mk
	and uncomment
	#include Makefile-GNU.mk
3) $make

/////////////////////////////////////////////////////////////////////////////
Required compile setup (Windows):
	Visual Studio 2008
	CUDA toolkit 2.3 or later
	CUDA SDK 2.3 or later
	Intel C++ Compiler 11.0.72 or later preferred

Required dependencies (Linux):
	CUDA 2.3 or later
	CUDA SDK 2.3 or later
	g++
	Intel C++ Compiler 11.1.56 or later preferred
	libglew-devel
	libgomp-devel (only when compiling with g++)
	freeglut 2.6.0 RC1 (freeglut.sourceforge.net) 
		freeglut 2.4.0 will not work

To set the needed environment variables in Windows,
run the following cmmands from an elevated prompt:

setx CUDA_SDK_INC_PATH "%NVSDKCUDA_ROOT%\common\inc"
setx CUDA_SDK_LIB_PATH "%NVSDKCUDA_ROOT%\common\lib"
setx CUDA_BIN_PATH_32 "%CUDA_BIN_PATH%\..\CUDA32\bin"
setx CUDA_LIB_PATH_32 "%CUDA_LIB_PATH%\..\CUDA32\lib"
/////////////////////////////////////////////////////////////////////////////
Other standard files:

StdAfx.h, StdAfx.cpp
    These files are used to build a precompiled header (PCH) file
    named ElectroMag.pch and a precompiled types file named StdAfx.obj.

/////////////////////////////////////////////////////////////////////////////
Known issues
- The "non-multithreaded" kernel is about 7 to 8 % faster than the "multithreaded" kernel 
 when the data size is a multiple of the number of multiprocesors.
- Under 64-bit mode, when not compiling with -maxrregcount 18, 19, or 20, the "multithreaded"
 kernel uses 21 registers, which reduces its throughput significantly.
- Other unknown issues may exist under 32-bit mode, as I did not intensivley test
 under 32-bit mode.
	


/////////////////////////////////////////////////////////////////////////////
Other notes:

AppWizard uses "TODO:" comments to indicate parts of the source code you
should add to or customize.

/////////////////////////////////////////////////////////////////////////////
