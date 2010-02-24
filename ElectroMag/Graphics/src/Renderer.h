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
// Generic macros for __declspec(dllexport). This modifier is needed if the symbols are to be
// visible ine the dll file under Windows.
#if (defined(_WIN64) || defined (_WIN32)) && defined (EMAG_BUILD)
#define EMAG_APIENTRY __declspec(dllexport)
// There is a SendMessage #define in <windows.h>. Very stupid on Microsoft's side
// If Microsoft were to follow good programming practice, it would be prefixed with something
// Anyway, we need to fix the problen, so we undefine this idiotic macro
#ifdef SendMessage
#undef SendMessage
#endif
#elif (defined(_WIN64) || defined (_WIN32))
#define EMAG_APIENTRY __declspec(dllimport)
#else
#define EMAG_APIENTRY
#endif
////////////////////////////////////////////////////////////////////////////////////////////////
///\defgroup GRAPHICS Graphics module used for data visualization
///@{
////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef _RENDERER_H
#define _RENDERER_H

namespace Render
{
/// Communication structure tat may be derived
struct RendererCommData
{
    void *commData;
};
class EMAG_APIENTRY Renderer	//abstract
{
public:
	//Renderer(){};
	//virtual ~Renderer(){};
	virtual void Init()=0;
	virtual void Draw()=0;
    /// Creates a new thread that starts the renderer asynchronously
    virtual void StartAsync() = 0;
    /// Kills the renderer thread
    virtual void KillAsync() = 0;
    /// Allows sending message and data to the renderer
    /// The exact form of the data is implementation-defined
    virtual void SendMessage(RendererCommData * messageData) = 0;
protected:  
    /// Pure function that can initialize the renderer asynchronously
    virtual void AsyncStartFunc() = 0;
};
}
#endif //_RENDERER_H
////////////////////////////////////////////////////////////////////////////////////////////////
///@}
////////////////////////////////////////////////////////////////////////////////////////////////

