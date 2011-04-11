================================================================================
    : ElectroMag Project Overview
================================================================================
Copyright Alexandru Gagniuc

================================================================================
    : Required packages
================================================================================
    gcc-c++ (use other compilers at your own risk)
    OpenMP devel library (libgomp or the omp that comes with your compiler)
Optional packages for CUDA:
    CUDA toolkit 2.3 or later
Optional packages for Graphics display
    glew devel 1.5 or higher
    freeglut devel 2.6.0  or higher
    Working OpenGL/GLU headers and libraries (you most definitely have these)

================================================================================
    : NOTES
================================================================================

1) Older versions of gcc
I have recieved several reports of compilation issues with gcc 4.1.2.
gcc 4.4.4 and later should work fine. Earlier versions have known issues about
some exotic c++ tricks which electromag uses.

2) Earlier versions of CMake
Some distributions package old versions of cmake. You may edit
CMakeLists.txt to select a lower cmake_minimum_required,
but your your milage may vary. The best option is to upgrade your cmake.

================================================================================
    : Compiling
================================================================================

0) cd to trunk directory
1) $ mkdir build
2) $ cd build
3) $ cmake ..
4) Optional: select Release build using ccmake ..
5) $ make
