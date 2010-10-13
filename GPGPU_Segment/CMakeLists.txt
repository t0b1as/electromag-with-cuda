#================================================
# CMake declarations for GPGPU_Segment
#================================================

# Host source files
set(GPGPU_SRCS
    src/Abstract_Functor.cpp
    src/CL_Electrostatics.cpp
    src/CL_Manager.cpp
    src/CUDA_Electrostatics.cpp
    src/CUDA_manager.cpp
    src/Electrostatics.cpp
    src/cuda_drvapi_dynlink.c
    src/OpenCL_Dyn_Load.c
)


    add_library( GPGPU_Segment STATIC
            ${GPGPU_SRCS})

#GPU source files
set(CUDA_SRCS
    src/Electrostatics.cu
    src/Electrostatics_Multistep.cu
)

find_package(CUDA)

IF(CUDA_FOUND)

    # Special thanks to James Bigler for the following suggestions

    # Compile the CUDA code to PTX. <my_target> is just a string used to set either the shared library flag <my_target>_EXPORTS and the generated file names' prefixes.
    CUDA_WRAP_SRCS(CUDA_PTX PTX generated_ptx_files ${CUDA_SRCS})

    # FindCUDA doesn't look for ptxas, but you can do it yourself:
    find_program(CUDA_PTXAS NAMES ptxas PATHS "${CUDA_TOOLKIT_ROOT_DIR}/bin")


    # Now set up the build rules to compile the PTX to CUBINs.
    set(generated_cubin_files)
    foreach(ptx_file ${generated_ptx_files})

        # You can get creative and use things like get_filename_component() to strip off the ptx from the filename.
        get_filename_component(PTX_NAME ${ptx_file} NAME_WE)
        set(generated_file "${PTX_NAME}.cubin")
        add_custom_command(
                OUTPUT ${generated_file}
                # These output files depend on the source_file and the contents of cmake_dependency_file
                MAIN_DEPENDENCY "${ptx_file}"
                # Here's the ptxas command
                COMMAND ${CUDA_PTXAS} "${ptx_file}" -o "${generated_file}"
                COMMENT "Generating ${generated_file}"
                )
        list(APPEND generated_cubin_files "${generated_file}")

    endforeach()

    CUDA_add_library(CUDA_LIB
                ${generated_cubin_files})


ELSE()
    MESSAGE("CUDA toolkit not found. Not building GPU code.")
ENDIF()
