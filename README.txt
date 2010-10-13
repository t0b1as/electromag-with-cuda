========================================================================
    : ElectroMag Project Overview
========================================================================.
Copyright Alexandru Gagniuc

Required packages:
    gcc-c++ (use other compilers at your own risk)
    OpenMP devel library (libgomp or the omp that comes with your compiler)
Optional packages for CUDA:
    CUDA toolkit 2.3 or later
Optional packages for Graphics display
    glew devel 1.5 or higher
    freeglut devel 2.6.0  or higher
    Working OpenGL/GLU headers and libraries (you most definitely have these)

========================================================================
    : Compiling
========================================================================.

0) cd to trunk directory
1) $ mkdir build
2) $ cd build
3) $ cmake ..
4) $ make



Known issues (Very moldy and dry pizza, do not waste your time reading this):
- The "non-multithreaded" kernel is about 7 to 8 % faster than the "multithreaded" kernel 
 when the data size is a multiple of the number of multiprocesors.
- Under 64-bit mode, when not compiling with -maxrregcount 18, 19, or 20, the "multithreaded"
 kernel uses 21 registers, which reduces its throughput significantly.
- With the 182.06 driver, non-paged to device memory copies seem to be capped around 3.3GB/s.
- Other unknown issues may exist under 32-bit mode, as I did not intensivley test
 under 32-bit mode.


