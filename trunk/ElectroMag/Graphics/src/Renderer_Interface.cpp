/***********************************************************************************************
Copyright (C) 2009-2010 - Alexandru Gagniuc - <http:\\g-tech.homeserver.com\HPC.htm>
 * This file is part of ElectroMag.

    ElectroMag is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    ElectroMag is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with ElectroMag.  If not, see <http://www.gnu.org/licenses/>.
***********************************************************************************************/
#include "GL/freeglut.h"
#include "Renderer Interface.h"

unsigned int GLRenderer::GlRenderers  = 0;
const unsigned int GLRenderer::maxGlRenderers = 2;
bool GLRenderer::glutIsInit = false;

void RendererInterface::StartAsync()
{
	unsigned long threadID;
    if(!rendererThread)
	{
        Threads::CreateNewThread((unsigned long (*)(void*))&RendererInterface::StartAsyncThreadFunc, this, &rendererThread, &threadID);
		Threads::SetThreadName(threadID, "Renderer");
	}
}

void RendererInterface::KillAsync()
{
    if(rendererThread)
        Threads::KillThread(rendererThread);
}

#include <iostream>

GLRenderer::GLRenderer()
{
	isActive = false;
}

void RendererInterface::StartAsyncThreadFunc(RendererInterface* objectToInit)
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

