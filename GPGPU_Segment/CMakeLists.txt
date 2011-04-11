#================================================
# CMake declarations for GPGPU_Segment
#================================================

# Host source files
set(GPGPU_SRCS
    src/Abstract_Functor.cpp
    src/CL_Electrostatics.cpp
    src/CL_Manager.cpp
    src/Electrostatics.cpp
    src/OpenCL_Dyn_Load.c
)


    add_library( GPGPU_Segment STATIC
            ${GPGPU_SRCS})
