/***********************************************************************************************
Copyright (C) 2010 - Alexandru Gagniuc - <http:\\g-tech.homeserver.com\HPC.htm>
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

#ifndef _FIELDRENDER_CLASS_H
#define _FIELDRENDER_CLASS_H

#include "Renderer Interface.h"
#include "FieldRender.h"
#include "Camera.h"
#include "Data Structures.h"
#include "Electrostatics.h"
#include <stdio.h>

enum ProjectionMode {Orthogonal, Perspective};
using namespace FieldRenderer;



class FieldRender: public GLRenderer
{
private:
    static FieldRenderer::GLpacket GLdata;
    static ProjectionMode PM;
    static Camera mainCam;
    static size_t lineSkip;
    // performance of calculations;
    static double perfGFLOP;
    void GLInit();
    /*static*/
    struct RenderData
    {
        static size_t bufferedLines;
    };

    // GL specific data
    bool VBOsupported;
    static unsigned int chargesVBO;
    static unsigned int colorVBO;
    static unsigned int *linesVBOs;
    static size_t nrLinesVBO;
    static float* colors;

    //Data about data
    bool dataBound;

    // signals that the renderes has finished
    static volatile bool quitFlag;

    void AsyncStartFunc();

    ///\brief Message reciever functor
    ///
    /// Overrides SendMessage from renderer, and passes the message to the message parser
    void SendMessage(Render::RendererCommData *message);

    ///\brief Message parser functor
    ///
    /// Parses external messages recieved by SendMessage(Render::RendererCommData *message)
    void SendMessage(FieldRenderer::FieldRenderCommData* message);

public:
    FieldRender();

    ~FieldRender();

    void Draw()
    {
    };

    void RenderPacket(FieldRenderer::GLpacket data)
    {
        this->dataBound = true;
        GLdata = data;
    }

    void SetLineSkip(size_t skip)
    {
        lineSkip = skip?skip:1;
    }

    void SetPerfGFLOP(double performance)
    {
        perfGFLOP = performance;
    }

    void Start();

private:

    unsigned long AsyncStartFunc(void * parameters);

    static void fieldDisplay();

    static void fieldDisplayVBO();

    static void reshape(int w, int h);

    static void keyboard(unsigned char key, int x, int y);

    static void mouse(int button, int state, int x, int y);

    static void motion(int x, int y);

    static void DrawOverlay();

};

#ifdef  __cplusplus
extern "C" {
#endif

    /// Factory functions for FieldRender
    EMAG_APIENTRY Render::Renderer* CreateFieldRenderer();
    EMAG_APIENTRY void DeleteFieldRenderer(Render::Renderer* objectToDelete);


#ifdef  __cplusplus
}
#endif

#endif//_FIELDRENDER_CLASS_H

