/***********************************************************************************************
Copyright (C) 2009 - Alexandru Gagniuc - <http:\\g-tech.homeserver.com\HPC.htm>
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

#include "freeglut_dynlink.h"


void* glutStrokeRoman;
void* glutStrokeMonoRoman;
void* glutBitmap9By15;
void* glutBitmap8By13;
void* glutBitmapTimesRoman10;
void* glutBitmapTimesRoman24;
void* glutBitmapHelvetica10;
void* glutBitmapHelvetica12;
void* glutBitmapHelvetica18;

__glutMainLoopEvent						*glutMainLoopEvent;
__glutLeaveMainLoop						*glutLeaveMainLoop;
__glutExit								*glutExit;
__glutInit								*glutInit;
__glutInitWindowPosition				*glutInitWindowPosition;
__glutInitWindowSize					*glutInitWindowSize;
__glutInitDisplayMode					*glutInitDisplayMode;
__glutInitDisplayString					*glutInitDisplayString;
__glutMainLoop							*glutMainLoop;
__glutCreateWindow						*glutCreateWindow;
__glutCreateSubWindow					*glutCreateSubWindow;
__glutDestroyWindow						*glutDestroyWindow;
__glutSetWindow							*glutSetWindow;
__glutGetWindow							*glutGetWindow;
__glutSetWindowTitle					*glutSetWindowTitle;
__glutSetIconTitle						*glutSetIconTitle;
__glutReshapeWindow						*glutReshapeWindow;
__glutPositionWindow					*glutPositionWindow;
__glutShowWindow						*glutShowWindow;
__glutHideWindow						*glutHideWindow;
__glutIconifyWindow						*glutIconifyWindow;
__glutPushWindow						*glutPushWindow;
__glutPopWindow							*glutPopWindow;
__glutFullScreen						*glutFullScreen;
__glutPostWindowRedisplay				*glutPostWindowRedisplay;
__glutPostRedisplay						*glutPostRedisplay;
__glutSwapBuffers						*glutSwapBuffers;
__glutWarpPointer						*glutWarpPointer;
__glutSetCursor							*glutSetCursor;
__glutEstablishOverlay					*glutEstablishOverlay;
__glutRemoveOverlay						*glutRemoveOverlay;
__glutUseLayer							*glutUseLayer;
__glutPostOverlayRedisplay				*glutPostOverlayRedisplay;
__glutPostWindowOverlayRedisplay		*glutPostWindowOverlayRedisplay;
__glutShowOverlay						*glutShowOverlay;
__glutHideOverlay						*glutHideOverlay;
__glutCreateMenu						*glutCreateMenu;
__glutDestroyMenu						*glutDestroyMenu;
__glutGetMenu							*glutGetMenu;
__glutSetMenu							*glutSetMenu;
__glutAddMenuEntry						*glutAddMenuEntry;
__glutAddSubMenu						*glutAddSubMenu;
__glutChangeToMenuEntry					*glutChangeToMenuEntry;
__glutChangeToSubMenu					*glutChangeToSubMenu;
__glutRemoveMenuItem					*glutRemoveMenuItem;
__glutAttachMenu						*glutAttachMenu;
__glutDetachMenu						*glutDetachMenu;
__glutTimerFunc							*glutTimerFunc;
__glutIdleFunc							*glutIdleFunc;
__glutKeyboardFunc						*glutKeyboardFunc;
__glutSpecialFunc						*glutSpecialFunc;
__glutReshapeFunc						*glutReshapeFunc;
__glutVisibilityFunc					*glutVisibilityFunc;
__glutDisplayFunc						*glutDisplayFunc;
__glutMouseFunc							*glutMouseFunc;
__glutMotionFunc						*glutMotionFunc;
__glutPassiveMotionFunc					*glutPassiveMotionFunc;
__glutEntryFunc							*glutEntryFunc;
__glutKeyboardUpFunc					*glutKeyboardUpFunc;
__glutSpecialUpFunc						*glutSpecialUpFunc;
__glutJoystickFunc						*glutJoystickFunc;
__glutMenuStateFunc						*glutMenuStateFunc;
__glutMenuStatusFunc					*glutMenuStatusFunc;
__glutOverlayDisplayFunc				*glutOverlayDisplayFunc;
__glutWindowStatusFunc					*glutWindowStatusFunc;
__glutSpaceballMotionFunc				*glutSpaceballMotionFunc;
__glutSpaceballRotateFunc				*glutSpaceballRotateFunc;
__glutSpaceballButtonFunc				*glutSpaceballButtonFunc;
__glutButtonBoxFunc						*glutButtonBoxFunc;
__glutDialsFunc							*glutDialsFunc;
__glutTabletMotionFunc					*glutTabletMotionFunc;
__glutTabletButtonFunc					*glutTabletButtonFunc;
__glutGet								*glutGet;
__glutDeviceGet							*glutDeviceGet;
__glutGetModifiers						*glutGetModifiers;
__glutLayerGet							*glutLayerGet;
__glutBitmapCharacter					*glutBitmapCharacter;
__glutBitmapWidth						*glutBitmapWidth;
__glutStrokeCharacter					*glutStrokeCharacter;
__glutStrokeWidth						*glutStrokeWidth;
__glutBitmapLength						*glutBitmapLength;
__glutStrokeLength						*glutStrokeLength;
__glutWireCube							*glutWireCube;
__glutSolidCube							*glutSolidCube;
__glutWireSphere						*glutWireSphere;
__glutSolidSphere						*glutSolidSphere;
__glutWireCone							*glutWireCone;
__glutSolidCone							*glutSolidCone;
__glutWireTorus							*glutWireTorus;
__glutSolidTorus						*glutSolidTorus;
__glutWireDodecahedron					*glutWireDodecahedron;
__glutSolidDodecahedron					*glutSolidDodecahedron;
__glutWireOctahedron					*glutWireOctahedron;
__glutSolidOctahedron					*glutSolidOctahedron;
__glutWireTetrahedron					*glutWireTetrahedron;
__glutSolidTetrahedron					*glutSolidTetrahedron;
__glutWireIcosahedron					*glutWireIcosahedron;
__glutSolidIcosahedron					*glutSolidIcosahedron;
__glutWireTeapot						*glutWireTeapot;
__glutSolidTeapot						*glutSolidTeapot;
__glutGameModeString					*glutGameModeString;
__glutEnterGameMode						*glutEnterGameMode;
__glutLeaveGameMode						*glutLeaveGameMode;
__glutGameModeGet						*glutGameModeGet;
__glutVideoResizeGet					*glutVideoResizeGet;
__glutSetupVideoResizing				*glutSetupVideoResizing;
__glutStopVideoResizing					*glutStopVideoResizing;
__glutVideoResize						*glutVideoResize;
__glutVideoPan							*glutVideoPan;
__glutSetColor							*glutSetColor;
__glutGetColor							*glutGetColor;
__glutCopyColormap						*glutCopyColormap;
__glutIgnoreKeyRepeat					*glutIgnoreKeyRepeat;
__glutSetKeyRepeat						*glutSetKeyRepeat;
__glutForceJoystickFunc					*glutForceJoystickFunc;
__glutExtensionSupported				*glutExtensionSupported;
__glutReportErrors						*glutReportErrors;
__glutFullScreenToggle					*glutFullScreenToggle;
__glutMouseWheelFunc					*glutMouseWheelFunc;
__glutCloseFunc							*glutCloseFunc;
__glutWMCloseFunc						*glutWMCloseFunc;
__glutMenuDestroyFunc					*glutMenuDestroyFunc;
__glutSetOption							*glutSetOption;
__glutGetModeValues						*glutGetModeValues;
__glutGetWindowData						*glutGetWindowData;
__glutSetWindowData						*glutSetWindowData;
__glutGetMenuData						*glutGetMenuData;
__glutSetMenuData						*glutSetMenuData;
__glutBitmapHeight						*glutBitmapHeight;
__glutStrokeHeight						*glutStrokeHeight;
__glutBitmapString						*glutBitmapString;
__glutStrokeString						*glutStrokeString;
__glutWireRhombicDodecahedron			*glutWireRhombicDodecahedron;
__glutSolidRhombicDodecahedron			*glutSolidRhombicDodecahedron;
__glutWireSierpinskiSponge				*glutWireSierpinskiSponge;
__glutSolidSierpinskiSponge				*glutSolidSierpinskiSponge;
__glutWireCylinder						*glutWireCylinder;
__glutSolidCylinder						*glutSolidCylinder;
__glutGetProcAddress					*glutGetProcAddress;
__glutInitContextVersion				*glutInitContextVersion;
__glutInitContextFlags					*glutInitContextFlags;
__glutInitContextProfile				*glutInitContextProfile;


////////////////////////////////////////////////////////////////////////////////////////////////
///
/// Here we define the library/symbol loading macros based on platform
///
////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(_WIN32) || defined(_WIN64)

    #include <Windows.h>

    #ifdef UNICODE
    static LPCWSTR __FgLibName = L"freeglut.dll";
    #else
    static LPCSTR __FgLibName = "freeglut.dll";
    #endif

    typedef HMODULE FG_LIB;

    enum FG_LibLoadCode LOAD_FG_LIBRARY(FG_LIB *pInstance)
    {
        *pInstance = LoadLibrary(__FgLibName);
        if (*pInstance == NULL)
        {
            return FG_ERROR_FILE_NOT_FOUND;
        }
        return FG_SUCCESS;
    }

    #define GET_PROC(name)                                          \
        name = (__##name *)GetProcAddress(CudaDrvLib, #name);        \
        if (name == NULL) return FG_ERROR_SYMBOL_NOT_FOUND

#elif defined(__unix__) || defined(__APPLE__) || defined(__MACOSX)

    #include <dlfcn.h>

    #if defined(__APPLE__) || defined(__MACOSX)
    static char __FgLibNameLocal[] = "libfreeglut.dylib";
    static char __FgLibNameLegacy[] = "libglut.dylib";
    static char __FgLibName[] = "/usr/lib/libfreeglut.dylib";
    #else
    static char __FgLibNameLocal[] = "libfreeglut.so";
    static char __FgLibNameLegacy[] = "libglut.so";
    static char __FgLibName[] = "/usr/lib/libfreeglut.so";
    #endif

    typedef void * FG_LIB;

    FG_LibLoadCode LOAD_FG_LIBRARY(FG_LIB *pInstance)
    {
        *pInstance = dlopen(__FgLibNameLocal, RTLD_NOW);
        if (*pInstance == NULL)
        {
            *pInstance = dlopen(__FgLibName, RTLD_NOW);
            if (*pInstance == NULL)
            {
                *pInstance = dlopen(__FgLibNameLegacy, RTLD_NOW);
                if (*pInstance == NULL)
                {
                    return FG_ERROR_FILE_NOT_FOUND;
                }
            }
        }
        return FG_SUCCESS;
    }

    #define GET_PROC(name)                                          \
        name = (__##name *)dlsym(CudaDrvLib, #name);                 \
        if (name == NULL) return FG_ERROR_SYMBOL_NOT_FOUND

#endif

FG_LibLoadCode glutLoadLibrary()
{
    FG_LIB CudaDrvLib;
    FG_LibLoadCode result;
    result = (LOAD_FG_LIBRARY(&CudaDrvLib));
	if(result != FG_SUCCESS)
	{
		return result;
	}

	GET_PROC(glutMainLoopEvent);
	GET_PROC(glutLeaveMainLoop);
	GET_PROC(glutExit);
	GET_PROC(glutInit);
	GET_PROC(glutInitWindowPosition);
	GET_PROC(glutInitWindowSize);
	GET_PROC(glutInitDisplayMode);
	GET_PROC(glutInitDisplayString);
	GET_PROC(glutMainLoop);
	GET_PROC(glutCreateWindow);
	GET_PROC(glutCreateSubWindow);
	GET_PROC(glutDestroyWindow);
	GET_PROC(glutSetWindow);
	GET_PROC(glutGetWindow);
	GET_PROC(glutSetWindowTitle);
	GET_PROC(glutSetIconTitle);
	GET_PROC(glutReshapeWindow);
	GET_PROC(glutPositionWindow);
	GET_PROC(glutShowWindow);
	GET_PROC(glutHideWindow);
	GET_PROC(glutIconifyWindow);
	GET_PROC(glutPushWindow);
	GET_PROC(glutPopWindow);
	GET_PROC(glutFullScreen);
	GET_PROC(glutPostWindowRedisplay);
	GET_PROC(glutPostRedisplay);
	GET_PROC(glutSwapBuffers);
	GET_PROC(glutWarpPointer);
	GET_PROC(glutSetCursor);
	GET_PROC(glutEstablishOverlay);
	GET_PROC(glutRemoveOverlay);
	GET_PROC(glutUseLayer);
	GET_PROC(glutPostOverlayRedisplay);
	GET_PROC(glutPostWindowOverlayRedisplay);
	GET_PROC(glutShowOverlay);
	GET_PROC(glutHideOverlay);
	GET_PROC(glutCreateMenu);
	GET_PROC(glutDestroyMenu);
	GET_PROC(glutGetMenu);
	GET_PROC(glutSetMenu);
	GET_PROC(glutAddMenuEntry);
	GET_PROC(glutAddSubMenu);
	GET_PROC(glutChangeToMenuEntry);
	GET_PROC(glutChangeToSubMenu);
	GET_PROC(glutRemoveMenuItem);
	GET_PROC(glutAttachMenu);
	GET_PROC(glutDetachMenu);
	GET_PROC(glutTimerFunc);
	GET_PROC(glutIdleFunc);
	GET_PROC(glutKeyboardFunc);
	GET_PROC(glutSpecialFunc);
	GET_PROC(glutReshapeFunc);
	GET_PROC(glutVisibilityFunc);
	GET_PROC(glutDisplayFunc);
	GET_PROC(glutMouseFunc);
	GET_PROC(glutMotionFunc);
	GET_PROC(glutPassiveMotionFunc);
	GET_PROC(glutEntryFunc);
	GET_PROC(glutKeyboardUpFunc);
	GET_PROC(glutSpecialUpFunc);
	GET_PROC(glutJoystickFunc);
	GET_PROC(glutMenuStateFunc);
	GET_PROC(glutMenuStatusFunc);
	GET_PROC(glutOverlayDisplayFunc);
	GET_PROC(glutWindowStatusFunc);
	GET_PROC(glutSpaceballMotionFunc);
	GET_PROC(glutSpaceballRotateFunc);
	GET_PROC(glutSpaceballButtonFunc);
	GET_PROC(glutButtonBoxFunc);
	GET_PROC(glutDialsFunc);
	GET_PROC(glutTabletMotionFunc);
	GET_PROC(glutTabletButtonFunc);
	GET_PROC(glutGet);
	GET_PROC(glutDeviceGet);
	GET_PROC(glutGetModifiers);
	GET_PROC(glutLayerGet);
	GET_PROC(glutBitmapCharacter);
	GET_PROC(glutBitmapWidth);
	GET_PROC(glutStrokeCharacter);
	GET_PROC(glutStrokeWidth);
	GET_PROC(glutBitmapLength);
	GET_PROC(glutStrokeLength);
	GET_PROC(glutWireCube);
	GET_PROC(glutSolidCube);
	GET_PROC(glutWireSphere);
	GET_PROC(glutSolidSphere);
	GET_PROC(glutWireCone);
	GET_PROC(glutSolidCone);
	GET_PROC(glutWireTorus);
	GET_PROC(glutSolidTorus);
	GET_PROC(glutWireDodecahedron);
	GET_PROC(glutSolidDodecahedron);
	GET_PROC(glutWireOctahedron);
	GET_PROC(glutSolidOctahedron);
	GET_PROC(glutWireTetrahedron);
	GET_PROC(glutSolidTetrahedron);
	GET_PROC(glutWireIcosahedron);
	GET_PROC(glutSolidIcosahedron);
	GET_PROC(glutWireTeapot);
	GET_PROC(glutSolidTeapot);
	GET_PROC(glutGameModeString);
	GET_PROC(glutEnterGameMode);
	GET_PROC(glutLeaveGameMode);
	GET_PROC(glutGameModeGet);
	GET_PROC(glutVideoResizeGet);
	GET_PROC(glutSetupVideoResizing);
	GET_PROC(glutStopVideoResizing);
	GET_PROC(glutVideoResize);
	GET_PROC(glutVideoPan);
	GET_PROC(glutSetColor);
	GET_PROC(glutGetColor);
	GET_PROC(glutCopyColormap);
	GET_PROC(glutIgnoreKeyRepeat);
	GET_PROC(glutSetKeyRepeat);
	GET_PROC(glutForceJoystickFunc);
	GET_PROC(glutExtensionSupported);
	GET_PROC(glutReportErrors);
	GET_PROC(glutFullScreenToggle);
	GET_PROC(glutMouseWheelFunc);
	GET_PROC(glutCloseFunc);
	GET_PROC(glutWMCloseFunc);
	GET_PROC(glutMenuDestroyFunc);
	GET_PROC(glutSetOption);
	GET_PROC(glutGetModeValues);
	GET_PROC(glutGetWindowData);
	GET_PROC(glutSetWindowData);
	GET_PROC(glutGetMenuData);
	GET_PROC(glutSetMenuData);
	GET_PROC(glutBitmapHeight);
	GET_PROC(glutStrokeHeight);
	GET_PROC(glutBitmapString);
	GET_PROC(glutStrokeString);
	GET_PROC(glutWireRhombicDodecahedron);
	GET_PROC(glutSolidRhombicDodecahedron);
	GET_PROC(glutWireSierpinskiSponge);
	GET_PROC(glutSolidSierpinskiSponge);
	GET_PROC(glutWireCylinder);
	GET_PROC(glutSolidCylinder);
	GET_PROC(glutGetProcAddress);
	GET_PROC(glutInitContextVersion);
	GET_PROC(glutInitContextFlags);
	GET_PROC(glutInitContextProfile);

    /*
    glutBitmapHelvetica12 = dlsym(CudaDrvLib, "glutBitmapHelvetica12");
    if(glutBitmapHelvetica12 == NULL)
        printf("Error in symbol. Err: %s\n", dlerror());
    else 
        printf("No Error\n");

    glutStrokeRoman = dlsym(CudaDrvLib, "glutStrokeRoman");
    glutStrokeMonoRoman = dlsym(CudaDrvLib, "glutStrokeMonoRoman");
    glutBitmap9By15 = dlsym(CudaDrvLib, "glutBitmap9By15");
    glutBitmap8By13 = dlsym(CudaDrvLib, "glutBitmap8By13");
    glutBitmapTimesRoman10 = dlsym(CudaDrvLib, "glutBitmapTimesRoman10");
    glutBitmapTimesRoman24 = dlsym(CudaDrvLib, "glutBitmapTimesRoman24");
    glutBitmapHelvetica10 = dlsym(CudaDrvLib, "glutBitmapHelvetica10");
    glutBitmapHelvetica12 = dlsym(CudaDrvLib, "glutBitmapHelvetica12");
    glutBitmapHelvetica18 = dlsym(CudaDrvLib, "glutBitmapHelvetica18");
     * */

    return FG_SUCCESS;
}
