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

add_library(EMagGraphics SHARED
    ${GRAPHICS_SRCS}
    )

target_link_libraries(EMagGraphics
    ${OPENGL_glu_LIBRARY} ${GLEW_LIBRARY} ${GLUT_glut_LIBRARY})