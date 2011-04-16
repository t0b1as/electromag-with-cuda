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
#ifndef _FIELDRENDER_H
#define _FIELDRENDER_H

#include <SOA_utils.hpp>
#include "Renderer.h"
#include "Electrostatics.h"

namespace FieldRenderer
{

struct GLpacket
{
    Array<electro::pointCharge<float> > *charges;
    Vector3<Array<float> > *lines;
    size_t nlines, lineLen;
    size_t elementSize;//8 for double 4 for float
};

enum MessageType
{
    /// Nothing is happening
    NoMessage = 0,
    /// *commData contains a GLpacket
    SendingGLData,
    /// *commData contains a double with the GFLOP/s pefomrmance
    SendingPerfPointer,
    /// we should put the address of shouldIQuit in commData
    RequestQuitFlag
};

struct FieldRenderCommData: public Render::RendererCommData
{
    MessageType messageType;
};
}



#endif  /* _FIELDRENDER_H */

