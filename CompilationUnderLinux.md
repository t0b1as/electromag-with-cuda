# Introduction #

In order to compile ElectroMag you will need GCC and cmake.


# Details #

ElectroMag does not explicitly depend on any proprietary blob. Any proprietary binary that ElectroMag uses is dynamically linked, so that ElectroMag may continue to work in entirely free systems. Therefore, you may still compile ElectroMag even if you do not have the proprietary NVIDIA drivers and/or the CUDA toolkit installed.
It will also be possible to use OpenCL on NVIDIA cards with the upcoming support from nouveau.

## Prerequisites ##
  1. GCC or other C++ compiler
  1. cmake
  1. Optionally, GL, GLU, GLEW, and GLUT developmental files to compile the graphics module
  1. Optionally, the CUDA runtime to compile the GPGPU code.

In the event you do not have the CUDA toolkit installed, the compilation will complete successfully, but GPU executable code will not be created. In that case, you may obtain the GPU modules from the downloads page. Make sure you get the version with the correct bitness for your build. Due to the design of CUDA, 32-bit modules are not interoperable with 64-bit builds of Electromag and vice-versa.Once you obtain the correct modles, simply copy them in the same folder with the ElectroMag executable.

Normally, all you have to do is cd to the directory and:

cmake .

make

### Using Graphics Support ###

ElectroMag supports displaying the results graphically. However, this requires freeglut, GLEW, GLU libraries to be available. As of [revision 32](https://code.google.com/p/electromag-with-cuda/source/detail?r=32), all graphics code has been moved into a separate module, which is dynamically loaded. ElectroMag may run without the graphics module, and it will still use OpenCL or CUDA acceleration if available.

To compile the graphics module, you will need the following developmental libraries:
  1. GLEW (preferably 1.5.2 or later, but earlier versions may work)
  1. GLU
  1. freeglut 2.6.0 headers are included in the source distribution

After compiling the Graphics module, you will need to create a symbolic link to "libEMagGraphics.so" (found in ./common/lib after compilation) in /usr/lib for the Graphics module to be loaded by ElectroMag. This is a limitation of how Linux handles dynamic linking, not a limitation of ElectroMag

# Using the Intel C++ compiler #

See cmake documentation