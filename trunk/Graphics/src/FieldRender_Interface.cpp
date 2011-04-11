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
#include "FieldRender.h"
#include "FieldRender class.h"

Render::Renderer* CreateFieldRenderer()
{
    return new FieldRender;
}

void DeleteFieldRenderer(Render::Renderer* objectToDelete)
{
    delete objectToDelete;
}

void FieldRender::SendMessage(Render::RendererCommData* message)
{
    this->SendMessage((FieldRenderer::FieldRenderCommData*) message);
}

void FieldRender::SendMessage(FieldRenderer::FieldRenderCommData* message)
{
    if (!message) return;
    switch (message->messageType)
    {
    case FieldRenderer::NoMessage:
        break;
    case FieldRenderer::SendingGLData:
        this->RenderPacket(*(FieldRenderer::GLpacket*)message->commData);
        break;
    case FieldRenderer::SendingPerfPointer:
        this->SetPerfGFLOP(*(double*)message->commData);
        break;
    case FieldRenderer::RequestQuitFlag:
        message->commData = (void*)&this->quitFlag;
        break;
    }

}
