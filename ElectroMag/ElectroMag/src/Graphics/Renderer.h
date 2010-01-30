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
#ifndef _RENDERER_H
#define _RENDERER_H
#include "X-Compat/Threading.h"

class Renderer	//abstract
{
public:
	Renderer(){rendererThread = 0;};
	virtual ~Renderer(){};
	virtual void Init()=0;
	virtual void Draw()=0;
    // Creates a new thread that starts the renderer asynchronously
    virtual void StartAsync();
    // Kills the renderer thread
    virtual void KillAsync();
protected:  
    // A handle to the thread handling the rendering
    Threads::ThreadHandle rendererThread;
    // Pure function that can initialize the renderer asynchronously
    virtual void AsyncStartFunc() = 0;
    // Static function that can be used to create a new thread
    static void StartAsyncThreadFunc(Renderer* objectToInit);
};

class GLRenderer : public Renderer //abstract
{
public:
	GLRenderer();
	virtual ~GLRenderer();
	virtual void Init();
protected:
	// openGL specific initialization function
	virtual void GLInit()=0;
	// Display function to be implemented in derived renderer classes
	//virtual static void Display()=0;
	/*/ Void input handling functions that can be overriden by derived classes
	virtual static void Keyboard(unsigned char key, int x, int y){};
	virtual static void Mouse(int button, int state, int x, int y){};
	virtual static void Reshape(int width, int height){};
	virtual static void Motion(int x, int y){};
	*/

	// 
	static bool glutLibIsLoaded;
private:
	static unsigned int  GlRenderers;
	static const unsigned int maxGlRenderers;
	bool isActive;
    static bool glutIsInit;
};

#endif //_RENDERER_H
