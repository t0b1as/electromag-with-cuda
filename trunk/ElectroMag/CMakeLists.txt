#================================================
# CMake declarations for ElectroMag
#================================================
set(ELECTROMAG_SRCS
    src/CPU_Implement.cpp
    src/ElectroMag.cpp
    src/Graphics_dynlink.cpp
    src/Particle_System.cpp
    src/stdafx.cpp
    src/CPUID/CPUID.cpp
)

add_executable(ElectroMag
    ${ELECTROMAG_SRCS}
    )

target_link_libraries(ElectroMag
                GPGPU_Segment  ${CMAKE_DL_LIBS} pthread)
