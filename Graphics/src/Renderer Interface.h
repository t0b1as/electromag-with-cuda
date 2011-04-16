/*
 * Copyright (C) 2010 - Alexandru Gagniuc - <mr.nuke.me@gmail.com>
 * This file is part of ElectroMag.
 *
 * ElectroMag is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * ElectroMag is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 *  along with ElectroMag.  If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef _RENDERER_INTERFACE_H
#define _RENDERER_INTERFACE_H

/**=============================================================================
 * \ingroup GRAPHICS Graphics module used for data visualization
 * @{
 * ===========================================================================*/

#include <thread>
#include "Renderer.h"

/**=============================================================================
 * \brief Renderer interface that defines abstract mechanisms for the Renderer
 * \brief class
 *
 * In order for programs to be able to dynamicaly link to this Graphics library
 * as a module, we need to be able to provide a fully abstract base class. This
 * way, external programs will be able to use a factory function to create an
 * instance of Renderer, and use the virtual function table to call functions
 * Since Renderer implements a few abstract mechanisms, we need to house those
 * to an interface class, and keep the Renderer class fully abstract. Thus,
 * those mechanisms are implemented in the Renderer interface class.
 * All renderer classes must derive from this class, not from the Renderer class
 * ===========================================================================*/
#include "Renderer.h"
class RendererInterface: public Render::Renderer
{
public:
    RendererInterface() {
        rendererThread = 0;
    };
    virtual ~RendererInterface() {};
    virtual void StartAsync();
    virtual void KillAsync();
    // This needs to be redefined here to ensure that dynamic linking on Windows
    // will not generate a pure virtual funvtion call error
    // How or why this happens is a mystery, but this fix seems to work.
    virtual void SendMessage(Render::RendererCommData * messageData) = 0;
protected:
    /// A handle to the thread handling the rendering
    std::thread *rendererThread;
    /// Static function that can be used to create a new thread
    static void StartAsyncThreadFunc(RendererInterface* objectToInit);
};

class GLRenderer : public RendererInterface
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
private:
    static unsigned int  GlRenderers;
    static const unsigned int maxGlRenderers;
    bool isActive;
    static bool glutIsInit;
};


#endif  /* _RENDERER_INTERFACE_H */

///@}
////////////////////////////////////////////////////////////////////////////////

