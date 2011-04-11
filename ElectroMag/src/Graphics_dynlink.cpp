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

#include "Graphics_dynlink.h"
#include <iostream>


namespace Graphics
{
__CreateFieldRenderer CreateFieldRenderer;
__DeleteFieldRenderer DeleteFieldRenderer;
}

#if defined(_WIN32) || defined(_WIN64)

#include <Windows.h>
using namespace  Graphics;

#ifdef UNICODE
static LPCWSTR __EmGraphLibName = L"Graphics.dll";
#else
static LPCSTR __EmGraphLibName = "Graphics.dll";
#endif

typedef HMODULE EM_GRAPH_LIB;

Graphics::ModuleLoadCode LoadEmGraphLib(EM_GRAPH_LIB *pInstance)
{
    *pInstance = LoadLibrary(__EmGraphLibName);
    if (*pInstance == NULL)
    {
        return Graphics::FILE_NOT_FOUND;
    }
    return Graphics::SUCCESS;
}

#define GET_PROC(name)                                          \
        name = (__##name)GetProcAddress(emGraphLib, #name);        \
        if (name == NULL) return Graphics::SYMBOL_NOT_FOUND

#elif defined(__unix__) || defined(__APPLE__) || defined(__MACOSX)

#include <dlfcn.h>
#if defined(__APPLE__) || defined(__MACOSX)
static char __EmGraphLibNameLocal[] = "libEMagGraphics.dylib";
static char __EmGraphLibName[] = "/usr/local/cuda/lib/libEMagGraphics.dylib";
#else
static char __EmGraphLibNameLocal[] = "./libEMagGraphics.so";
static char __EmGraphLibName[] = "libEMagGraphics.so";
#endif

typedef void * EM_GRAPH_LIB;

Graphics::ModuleLoadCode LoadEmGraphLib(EM_GRAPH_LIB *pInstance)
{
    *pInstance = dlopen(__EmGraphLibNameLocal, RTLD_NOW);
    if (*pInstance == 0)
    {
        *pInstance = dlopen(__EmGraphLibName, RTLD_NOW);
        if (*pInstance == 0)
        {
            std::cerr<<dlerror()<<std::endl;
            return Graphics::FILE_NOT_FOUND;
        }
    }
    return Graphics::SUCCESS;
}

#define GET_PROC(name)                                          \
        name = (__##name)(size_t)dlsym(emGraphLib, #name);                 \
        if (name == 0) {std::cerr<<dlerror()<<std::endl;return SYMBOL_NOT_FOUND;}

#endif

namespace Graphics
{
ModuleLoadCode LoadModule()
{
    EM_GRAPH_LIB emGraphLib;
    ModuleLoadCode errCode;

    errCode = LoadEmGraphLib(&emGraphLib);
    if (errCode != SUCCESS)
    {
        return errCode;
    }
    GET_PROC(CreateFieldRenderer);
    GET_PROC(DeleteFieldRenderer);

    return SUCCESS;
}
}
