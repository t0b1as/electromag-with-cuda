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


#ifndef _GRAPHICS_DYNLINK_H
#define	_GRAPHICS_DYNLINK_H
// We won't need the declaration of Renderer if all we are doing is dynamically link
#include "../../Graphics/src/FieldRender.h"

namespace Graphics
{

typedef Render::Renderer* (*__CreateFieldRenderer)();
typedef void (*__DeleteFieldRenderer)(Render::Renderer* objectToDelete);

extern __CreateFieldRenderer CreateFieldRenderer;
extern __DeleteFieldRenderer DeleteFieldRenderer;

enum ModuleLoadCode
{
    SUCCESS = 0,
    FILE_NOT_FOUND,
    SYMBOL_NOT_FOUND
};

ModuleLoadCode LoadModule();
}
#endif	/* _GRAPHICS_DYNLINK_H */

