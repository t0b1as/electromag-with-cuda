#include "GL/freeglut.h"
#include "Renderer.h"

unsigned int GLRenderer::GlRenderers  = 0;
const unsigned int GLRenderer::maxGlRenderers = 2;
bool GLRenderer::glutIsInit = false;

void Renderer::StartAsync()
{
    if(!rendererThread)
        CreateNewThread((unsigned long (*)(void*))&Renderer::StartAsyncThreadFunc, this, &rendererThread);
}

void Renderer::KillAsync()
{
    if(rendererThread)
        KillThread(rendererThread);
}
GLRenderer::GLRenderer()
{
	isActive = false;
}

void Renderer::StartAsyncThreadFunc(Renderer* objectToInit)
{
    objectToInit->AsyncStartFunc();
}

GLRenderer::~GLRenderer()
{
	if(isActive)
	{
		GlRenderers--;
		isActive = false;
	}
}
                                                                                                                        
void GLRenderer::Init()
{
	if(isActive) return; // Do nothing

	if(GlRenderers >= maxGlRenderers)
	{
		throw(" Only one active OpenGL renderer allowed");
		return;
	}

    if(!glutIsInit)
    {
        int zero = 0;
        glutInit(&zero, 0);
        glutIsInit = true;
    }
    
	GlRenderers ++;
	isActive = true;
	GLInit();

	// Register display and inputhandling functions
	/*glutDisplayFunc(Display);
	glutDisplayFunc(Keyboard);
	glutDisplayFunc(Mouse);
	glutDisplayFunc(Reshape);
	glutDisplayFunc(Motion);
	*/
}
