#================================================
# CMake declarations for Graphics
#================================================

set(GRAPHICS_SRCS
    src/FieldRender.cpp
    src/FieldRender_Interface.cpp
    src/FrontendGUI.cpp
    src/MainGUI.cpp
    src/Renderer_Interface.cpp
)

add_library(libEMagGraphics SHARED
    ${GRAPHICS_SRCS}
    )