/***********************************************************************************************
Copyright (C) 2009-2010 - Alexandru Gagniuc - <http:\\g-tech.homeserver.com\HPC.htm>
Contains source originally written by Pawel W. Olszta, <olszta@sourceforge.net>

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

#ifndef  _FREEGLUT_DYNLINK_H
#define  _FREEGLUT_DYNLINK_H

#ifdef __cplusplus
    extern "C" {
#endif

/*
 * Under windows, we have to differentiate between static and dynamic libraries
 */
#ifdef _WIN32
/* #pragma may not be supported by some compilers.
 * Discussion by FreeGLUT developers suggests that
 * Visual C++ specific code involving pragmas may
 * need to move to a separate header.  24th Dec 2003
 */

/* Define FREEGLUT_LIB_PRAGMAS to 1 to include library
 * pragmas or to 1 to exclude library pragmas.
 * The default behavior depends on the compiler/platform.
 */
#   ifndef FREEGLUT_LIB_PRAGMAS
#       if ( defined(_MSC_VER) || defined(__WATCOMC__) ) && !defined(_WIN32_WCE)
#           define FREEGLUT_LIB_PRAGMAS 1
#			pragma comment (lib, "glu32.lib")    /* link OpenGL Utility lib     */
#		    pragma comment (lib, "opengl32.lib") /* link Microsoft OpenGL lib   */
//#			pragma comment (lib, "gdi32.lib")    /* link Windows GDI lib        */
//#		    pragma comment (lib, "winmm.lib")    /* link Windows MultiMedia lib */
//#			pragma comment (lib, "user32.lib")   /* link Windows user lib       */
#       else
#           define FREEGLUT_LIB_PRAGMAS 0
#       endif
#   endif

/* Windows static library */
#   ifdef FREEGLUT_STATIC

#       define FGAPI
#       define FGAPIENTRY

        /* Link with Win32 static freeglut lib */
#       if FREEGLUT_LIB_PRAGMAS
#           pragma comment (lib, "freeglut_static.lib")
#       endif

/* Windows shared library (DLL) */
#   else

#       define FGAPIENTRY __stdcall
#       if defined(FREEGLUT_EXPORTS)
#           define FGAPI __declspec(dllexport)
#       else
#           define FGAPI __declspec(dllimport)
#       endif

#   endif

#else

/* Non-Windows definition of FGAPI and FGAPIENTRY  */
#        define FGAPI
#        define FGAPIENTRY

#endif

/*
 * The freeglut and GLUT API versions
 */
#define  FREEGLUT             1
#define  GLUT_API_VERSION     4
#define  FREEGLUT_VERSION_2_0 1
#define  GLUT_XLIB_IMPLEMENTATION 13

/*
 * Always include OpenGL and GLU headers
 */

//#pragma message("these headers may not need to ve included if linking dynamically at runtime")
// We include GLEW to prevent name pollution. GLEW includes both basic gl definitions and glu.h
#include "glew_static.h"


////////////////////////////////////////////////////////////////////////////////////////////////
///\brief freeglut_std.h defines
///
////////////////////////////////////////////////////////////////////////////////////////////////

/*
 * GLUT API macro definitions -- the special key codes:
 */
#define  GLUT_KEY_F1                        0x0001
#define  GLUT_KEY_F2                        0x0002
#define  GLUT_KEY_F3                        0x0003
#define  GLUT_KEY_F4                        0x0004
#define  GLUT_KEY_F5                        0x0005
#define  GLUT_KEY_F6                        0x0006
#define  GLUT_KEY_F7                        0x0007
#define  GLUT_KEY_F8                        0x0008
#define  GLUT_KEY_F9                        0x0009
#define  GLUT_KEY_F10                       0x000A
#define  GLUT_KEY_F11                       0x000B
#define  GLUT_KEY_F12                       0x000C
#define  GLUT_KEY_LEFT                      0x0064
#define  GLUT_KEY_UP                        0x0065
#define  GLUT_KEY_RIGHT                     0x0066
#define  GLUT_KEY_DOWN                      0x0067
#define  GLUT_KEY_PAGE_UP                   0x0068
#define  GLUT_KEY_PAGE_DOWN                 0x0069
#define  GLUT_KEY_HOME                      0x006A
#define  GLUT_KEY_END                       0x006B
#define  GLUT_KEY_INSERT                    0x006C

/*
 * GLUT API macro definitions -- mouse state definitions
 */
#define  GLUT_LEFT_BUTTON                   0x0000
#define  GLUT_MIDDLE_BUTTON                 0x0001
#define  GLUT_RIGHT_BUTTON                  0x0002
#define  GLUT_DOWN                          0x0000
#define  GLUT_UP                            0x0001
#define  GLUT_LEFT                          0x0000
#define  GLUT_ENTERED                       0x0001

/*
 * GLUT API macro definitions -- the display mode definitions
 */
#define  GLUT_RGB                           0x0000
#define  GLUT_RGBA                          0x0000
#define  GLUT_INDEX                         0x0001
#define  GLUT_SINGLE                        0x0000
#define  GLUT_DOUBLE                        0x0002
#define  GLUT_ACCUM                         0x0004
#define  GLUT_ALPHA                         0x0008
#define  GLUT_DEPTH                         0x0010
#define  GLUT_STENCIL                       0x0020
#define  GLUT_MULTISAMPLE                   0x0080
#define  GLUT_STEREO                        0x0100
#define  GLUT_LUMINANCE                     0x0200

/*
 * GLUT API macro definitions -- windows and menu related definitions
 */
#define  GLUT_MENU_NOT_IN_USE               0x0000
#define  GLUT_MENU_IN_USE                   0x0001
#define  GLUT_NOT_VISIBLE                   0x0000
#define  GLUT_VISIBLE                       0x0001
#define  GLUT_HIDDEN                        0x0000
#define  GLUT_FULLY_RETAINED                0x0001
#define  GLUT_PARTIALLY_RETAINED            0x0002
#define  GLUT_FULLY_COVERED                 0x0003

/*
 * GLUT API macro definitions -- fonts definitions
 *
 * Steve Baker suggested to make it binary compatible with GLUT:
 */
#if defined(_MSC_VER) || defined(__CYGWIN__) || defined(__MINGW32__) || defined(__WATCOMC__)
#   define  GLUT_STROKE_ROMAN               ((void *)0x0000)
#   define  GLUT_STROKE_MONO_ROMAN          ((void *)0x0001)
#   define  GLUT_BITMAP_9_BY_15             ((void *)0x0002)
#   define  GLUT_BITMAP_8_BY_13             ((void *)0x0003)
#   define  GLUT_BITMAP_TIMES_ROMAN_10      ((void *)0x0004)
#   define  GLUT_BITMAP_TIMES_ROMAN_24      ((void *)0x0005)
#   define  GLUT_BITMAP_HELVETICA_10        ((void *)0x0006)
#   define  GLUT_BITMAP_HELVETICA_12        ((void *)0x0007)
#   define  GLUT_BITMAP_HELVETICA_18        ((void *)0x0008)
#else
    /*
     * I don't really know if it's a good idea... But here it goes:
     */
    extern void* glutStrokeRoman;
    extern void* glutStrokeMonoRoman;
    extern void* glutBitmap9By15;
    extern void* glutBitmap8By13;
    extern void* glutBitmapTimesRoman10;
    extern void* glutBitmapTimesRoman24;
    extern void* glutBitmapHelvetica10;
    extern void* glutBitmapHelvetica12;
    extern void* glutBitmapHelvetica18;

    /*
     * Those pointers will be used by following definitions:
     */
#   define  GLUT_STROKE_ROMAN               ((void *) &glutStrokeRoman)
#   define  GLUT_STROKE_MONO_ROMAN          ((void *) &glutStrokeMonoRoman)
#   define  GLUT_BITMAP_9_BY_15             ((void *) &glutBitmap9By15)
#   define  GLUT_BITMAP_8_BY_13             ((void *) &glutBitmap8By13)
#   define  GLUT_BITMAP_TIMES_ROMAN_10      ((void *) &glutBitmapTimesRoman10)
#   define  GLUT_BITMAP_TIMES_ROMAN_24      ((void *) &glutBitmapTimesRoman24)
#   define  GLUT_BITMAP_HELVETICA_10        ((void *) &glutBitmapHelvetica10)
#   define  GLUT_BITMAP_HELVETICA_12        ((void *) &glutBitmapHelvetica12)
#   define  GLUT_BITMAP_HELVETICA_18        ((void *) &glutBitmapHelvetica18)
#endif

/*
 * GLUT API macro definitions -- the glutGet parameters
 */
#define  GLUT_WINDOW_X                      0x0064
#define  GLUT_WINDOW_Y                      0x0065
#define  GLUT_WINDOW_WIDTH                  0x0066
#define  GLUT_WINDOW_HEIGHT                 0x0067
#define  GLUT_WINDOW_BUFFER_SIZE            0x0068
#define  GLUT_WINDOW_STENCIL_SIZE           0x0069
#define  GLUT_WINDOW_DEPTH_SIZE             0x006A
#define  GLUT_WINDOW_RED_SIZE               0x006B
#define  GLUT_WINDOW_GREEN_SIZE             0x006C
#define  GLUT_WINDOW_BLUE_SIZE              0x006D
#define  GLUT_WINDOW_ALPHA_SIZE             0x006E
#define  GLUT_WINDOW_ACCUM_RED_SIZE         0x006F
#define  GLUT_WINDOW_ACCUM_GREEN_SIZE       0x0070
#define  GLUT_WINDOW_ACCUM_BLUE_SIZE        0x0071
#define  GLUT_WINDOW_ACCUM_ALPHA_SIZE       0x0072
#define  GLUT_WINDOW_DOUBLEBUFFER           0x0073
#define  GLUT_WINDOW_RGBA                   0x0074
#define  GLUT_WINDOW_PARENT                 0x0075
#define  GLUT_WINDOW_NUM_CHILDREN           0x0076
#define  GLUT_WINDOW_COLORMAP_SIZE          0x0077
#define  GLUT_WINDOW_NUM_SAMPLES            0x0078
#define  GLUT_WINDOW_STEREO                 0x0079
#define  GLUT_WINDOW_CURSOR                 0x007A

#define  GLUT_SCREEN_WIDTH                  0x00C8
#define  GLUT_SCREEN_HEIGHT                 0x00C9
#define  GLUT_SCREEN_WIDTH_MM               0x00CA
#define  GLUT_SCREEN_HEIGHT_MM              0x00CB
#define  GLUT_MENU_NUM_ITEMS                0x012C
#define  GLUT_DISPLAY_MODE_POSSIBLE         0x0190
#define  GLUT_INIT_WINDOW_X                 0x01F4
#define  GLUT_INIT_WINDOW_Y                 0x01F5
#define  GLUT_INIT_WINDOW_WIDTH             0x01F6
#define  GLUT_INIT_WINDOW_HEIGHT            0x01F7
#define  GLUT_INIT_DISPLAY_MODE             0x01F8
#define  GLUT_ELAPSED_TIME                  0x02BC
#define  GLUT_WINDOW_FORMAT_ID              0x007B

/*
 * GLUT API macro definitions -- the glutDeviceGet parameters
 */
#define  GLUT_HAS_KEYBOARD                  0x0258
#define  GLUT_HAS_MOUSE                     0x0259
#define  GLUT_HAS_SPACEBALL                 0x025A
#define  GLUT_HAS_DIAL_AND_BUTTON_BOX       0x025B
#define  GLUT_HAS_TABLET                    0x025C
#define  GLUT_NUM_MOUSE_BUTTONS             0x025D
#define  GLUT_NUM_SPACEBALL_BUTTONS         0x025E
#define  GLUT_NUM_BUTTON_BOX_BUTTONS        0x025F
#define  GLUT_NUM_DIALS                     0x0260
#define  GLUT_NUM_TABLET_BUTTONS            0x0261
#define  GLUT_DEVICE_IGNORE_KEY_REPEAT      0x0262
#define  GLUT_DEVICE_KEY_REPEAT             0x0263
#define  GLUT_HAS_JOYSTICK                  0x0264
#define  GLUT_OWNS_JOYSTICK                 0x0265
#define  GLUT_JOYSTICK_BUTTONS              0x0266
#define  GLUT_JOYSTICK_AXES                 0x0267
#define  GLUT_JOYSTICK_POLL_RATE            0x0268

/*
 * GLUT API macro definitions -- the glutLayerGet parameters
 */
#define  GLUT_OVERLAY_POSSIBLE              0x0320
#define  GLUT_LAYER_IN_USE                  0x0321
#define  GLUT_HAS_OVERLAY                   0x0322
#define  GLUT_TRANSPARENT_INDEX             0x0323
#define  GLUT_NORMAL_DAMAGED                0x0324
#define  GLUT_OVERLAY_DAMAGED               0x0325

/*
 * GLUT API macro definitions -- the glutVideoResizeGet parameters
 */
#define  GLUT_VIDEO_RESIZE_POSSIBLE         0x0384
#define  GLUT_VIDEO_RESIZE_IN_USE           0x0385
#define  GLUT_VIDEO_RESIZE_X_DELTA          0x0386
#define  GLUT_VIDEO_RESIZE_Y_DELTA          0x0387
#define  GLUT_VIDEO_RESIZE_WIDTH_DELTA      0x0388
#define  GLUT_VIDEO_RESIZE_HEIGHT_DELTA     0x0389
#define  GLUT_VIDEO_RESIZE_X                0x038A
#define  GLUT_VIDEO_RESIZE_Y                0x038B
#define  GLUT_VIDEO_RESIZE_WIDTH            0x038C
#define  GLUT_VIDEO_RESIZE_HEIGHT           0x038D

/*
 * GLUT API macro definitions -- the glutUseLayer parameters
 */
#define  GLUT_NORMAL                        0x0000
#define  GLUT_OVERLAY                       0x0001

/*
 * GLUT API macro definitions -- the glutGetModifiers parameters
 */
#define  GLUT_ACTIVE_SHIFT                  0x0001
#define  GLUT_ACTIVE_CTRL                   0x0002
#define  GLUT_ACTIVE_ALT                    0x0004

/*
 * GLUT API macro definitions -- the glutSetCursor parameters
 */
#define  GLUT_CURSOR_RIGHT_ARROW            0x0000
#define  GLUT_CURSOR_LEFT_ARROW             0x0001
#define  GLUT_CURSOR_INFO                   0x0002
#define  GLUT_CURSOR_DESTROY                0x0003
#define  GLUT_CURSOR_HELP                   0x0004
#define  GLUT_CURSOR_CYCLE                  0x0005
#define  GLUT_CURSOR_SPRAY                  0x0006
#define  GLUT_CURSOR_WAIT                   0x0007
#define  GLUT_CURSOR_TEXT                   0x0008
#define  GLUT_CURSOR_CROSSHAIR              0x0009
#define  GLUT_CURSOR_UP_DOWN                0x000A
#define  GLUT_CURSOR_LEFT_RIGHT             0x000B
#define  GLUT_CURSOR_TOP_SIDE               0x000C
#define  GLUT_CURSOR_BOTTOM_SIDE            0x000D
#define  GLUT_CURSOR_LEFT_SIDE              0x000E
#define  GLUT_CURSOR_RIGHT_SIDE             0x000F
#define  GLUT_CURSOR_TOP_LEFT_CORNER        0x0010
#define  GLUT_CURSOR_TOP_RIGHT_CORNER       0x0011
#define  GLUT_CURSOR_BOTTOM_RIGHT_CORNER    0x0012
#define  GLUT_CURSOR_BOTTOM_LEFT_CORNER     0x0013
#define  GLUT_CURSOR_INHERIT                0x0064
#define  GLUT_CURSOR_NONE                   0x0065
#define  GLUT_CURSOR_FULL_CROSSHAIR         0x0066

/*
 * GLUT API macro definitions -- RGB color component specification definitions
 */
#define  GLUT_RED                           0x0000
#define  GLUT_GREEN                         0x0001
#define  GLUT_BLUE                          0x0002

/*
 * GLUT API macro definitions -- additional keyboard and joystick definitions
 */
#define  GLUT_KEY_REPEAT_OFF                0x0000
#define  GLUT_KEY_REPEAT_ON                 0x0001
#define  GLUT_KEY_REPEAT_DEFAULT            0x0002

#define  GLUT_JOYSTICK_BUTTON_A             0x0001
#define  GLUT_JOYSTICK_BUTTON_B             0x0002
#define  GLUT_JOYSTICK_BUTTON_C             0x0004
#define  GLUT_JOYSTICK_BUTTON_D             0x0008

/*
 * GLUT API macro definitions -- game mode definitions
 */
#define  GLUT_GAME_MODE_ACTIVE              0x0000
#define  GLUT_GAME_MODE_POSSIBLE            0x0001
#define  GLUT_GAME_MODE_WIDTH               0x0002
#define  GLUT_GAME_MODE_HEIGHT              0x0003
#define  GLUT_GAME_MODE_PIXEL_DEPTH         0x0004
#define  GLUT_GAME_MODE_REFRESH_RATE        0x0005
#define  GLUT_GAME_MODE_DISPLAY_CHANGED     0x0006

////////////////////////////////////////////////////////////////////////////////////////////////
///\brief freeglut_ext.h defines
///
////////////////////////////////////////////////////////////////////////////////////////////////

/*
 * Additional GLUT Key definitions for the Special key function
 */
#define GLUT_KEY_NUM_LOCK           0x006D
#define GLUT_KEY_BEGIN              0x006E
#define GLUT_KEY_DELETE             0x006F

/*
 * GLUT API Extension macro definitions -- behaviour when the user clicks on an "x" to close a window
 */
#define GLUT_ACTION_EXIT                         0
#define GLUT_ACTION_GLUTMAINLOOP_RETURNS         1
#define GLUT_ACTION_CONTINUE_EXECUTION           2

/*
 * Create a new rendering context when the user opens a new window?
 */
#define GLUT_CREATE_NEW_CONTEXT                  0
#define GLUT_USE_CURRENT_CONTEXT                 1

/*
 * Direct/Indirect rendering context options (has meaning only in Unix/X11)
 */
#define GLUT_FORCE_INDIRECT_CONTEXT              0
#define GLUT_ALLOW_DIRECT_CONTEXT                1
#define GLUT_TRY_DIRECT_CONTEXT                  2
#define GLUT_FORCE_DIRECT_CONTEXT                3

/*
 * GLUT API Extension macro definitions -- the glutGet parameters
 */
#define  GLUT_INIT_STATE                    0x007C

#define  GLUT_ACTION_ON_WINDOW_CLOSE        0x01F9

#define  GLUT_WINDOW_BORDER_WIDTH           0x01FA
#define  GLUT_WINDOW_HEADER_HEIGHT          0x01FB

#define  GLUT_VERSION                       0x01FC

#define  GLUT_RENDERING_CONTEXT             0x01FD
#define  GLUT_DIRECT_RENDERING              0x01FE

#define  GLUT_FULL_SCREEN                   0x01FF

/*
 * New tokens for glutInitDisplayMode.
 * Only one GLUT_AUXn bit may be used at a time.
 * Value 0x0400 is defined in OpenGLUT.
 */
#define  GLUT_AUX                           0x1000

#define  GLUT_AUX1                          0x1000
#define  GLUT_AUX2                          0x2000
#define  GLUT_AUX3                          0x4000
#define  GLUT_AUX4                          0x8000

/*
 * Context-related flags, see freeglut_state.c
 */
#define  GLUT_INIT_MAJOR_VERSION            0x0200
#define  GLUT_INIT_MINOR_VERSION            0x0201
#define  GLUT_INIT_FLAGS                    0x0202
#define  GLUT_INIT_PROFILE                  0x0203

/*
 * Flags for glutInitContextFlags, see freeglut_init.c
 */
#define  GLUT_DEBUG                         0x0001
#define  GLUT_FORWARD_COMPATIBLE            0x0002


/*
 * Flags for glutInitContextProfile, see freeglut_init.c
 */
#define GLUT_CORE_PROFILE                   0x0001
#define	GLUT_COMPATIBILITY_PROFILE          0x0002

/*
 * Process loop function, see free__glut_main.c
 */
typedef void    FGAPIENTRY __glutMainLoopEvent( void );
typedef void    FGAPIENTRY __glutLeaveMainLoop( void );
typedef void    FGAPIENTRY __glutExit         ( void );


////////////////////////////////////////////////////////////////////////////////////////////////
///\brief free__glut_std.h functions
///
////////////////////////////////////////////////////////////////////////////////////////////////


/*
 * Initialization functions, see f__glut_init.c
 */
typedef void    FGAPIENTRY __glutInit( int* pargc, char** argv );
typedef void    FGAPIENTRY __glutInitWindowPosition( int x, int y );
typedef void    FGAPIENTRY __glutInitWindowSize( int width, int height );
typedef void    FGAPIENTRY __glutInitDisplayMode( unsigned int displayMode );
typedef void    FGAPIENTRY __glutInitDisplayString( const char* displayMode );

/*
 * Process loop function, see free__glut_main.c
 */
typedef void    FGAPIENTRY __glutMainLoop( void );

/*
 * Window management functions, see free__glut_window.c
 */
typedef int     FGAPIENTRY __glutCreateWindow( const char* title );
typedef int     FGAPIENTRY __glutCreateSubWindow( int window, int x, int y, int width, int height );
typedef void    FGAPIENTRY __glutDestroyWindow( int window );
typedef void    FGAPIENTRY __glutSetWindow( int window );
typedef int     FGAPIENTRY __glutGetWindow( void );
typedef void    FGAPIENTRY __glutSetWindowTitle( const char* title );
typedef void    FGAPIENTRY __glutSetIconTitle( const char* title );
typedef void    FGAPIENTRY __glutReshapeWindow( int width, int height );
typedef void    FGAPIENTRY __glutPositionWindow( int x, int y );
typedef void    FGAPIENTRY __glutShowWindow( void );
typedef void    FGAPIENTRY __glutHideWindow( void );
typedef void    FGAPIENTRY __glutIconifyWindow( void );
typedef void    FGAPIENTRY __glutPushWindow( void );
typedef void    FGAPIENTRY __glutPopWindow( void );
typedef void    FGAPIENTRY __glutFullScreen( void );

/*
 * Display-connected functions, see free__glut_display.c
 */
typedef void    FGAPIENTRY __glutPostWindowRedisplay( int window );
typedef void    FGAPIENTRY __glutPostRedisplay( void );
typedef void    FGAPIENTRY __glutSwapBuffers( void );

/*
 * Mouse cursor functions, see free__glut_cursor.c
 */
typedef void    FGAPIENTRY __glutWarpPointer( int x, int y );
typedef void    FGAPIENTRY __glutSetCursor( int cursor );

/*
 * Overlay stuff, see free__glut_overlay.c
 */
typedef void    FGAPIENTRY __glutEstablishOverlay( void );
typedef void    FGAPIENTRY __glutRemoveOverlay( void );
typedef void    FGAPIENTRY __glutUseLayer( GLenum layer );
typedef void    FGAPIENTRY __glutPostOverlayRedisplay( void );
typedef void    FGAPIENTRY __glutPostWindowOverlayRedisplay( int window );
typedef void    FGAPIENTRY __glutShowOverlay( void );
typedef void    FGAPIENTRY __glutHideOverlay( void );

/*
 * Menu stuff, see free__glut_menu.c
 */
typedef int     FGAPIENTRY __glutCreateMenu( void (* callback)( int menu ) );
typedef void    FGAPIENTRY __glutDestroyMenu( int menu );
typedef int     FGAPIENTRY __glutGetMenu( void );
typedef void    FGAPIENTRY __glutSetMenu( int menu );
typedef void    FGAPIENTRY __glutAddMenuEntry( const char* label, int value );
typedef void    FGAPIENTRY __glutAddSubMenu( const char* label, int subMenu );
typedef void    FGAPIENTRY __glutChangeToMenuEntry( int item, const char* label, int value );
typedef void    FGAPIENTRY __glutChangeToSubMenu( int item, const char* label, int value );
typedef void    FGAPIENTRY __glutRemoveMenuItem( int item );
typedef void    FGAPIENTRY __glutAttachMenu( int button );
typedef void    FGAPIENTRY __glutDetachMenu( int button );

/*
 * Global callback functions, see free__glut_callbacks.c
 */
typedef void    FGAPIENTRY __glutTimerFunc( unsigned int time, void (* callback)( int ), int value );
typedef void    FGAPIENTRY __glutIdleFunc( void (* callback)( void ) );

/*
 * Window-specific callback functions, see free__glut_callbacks.c
 */
typedef void    FGAPIENTRY __glutKeyboardFunc( void (* callback)( unsigned char, int, int ) );
typedef void    FGAPIENTRY __glutSpecialFunc( void (* callback)( int, int, int ) );
typedef void    FGAPIENTRY __glutReshapeFunc( void (* callback)( int, int ) );
typedef void    FGAPIENTRY __glutVisibilityFunc( void (* callback)( int ) );
typedef void    FGAPIENTRY __glutDisplayFunc( void (* callback)( void ) );
typedef void    FGAPIENTRY __glutMouseFunc( void (* callback)( int, int, int, int ) );
typedef void    FGAPIENTRY __glutMotionFunc( void (* callback)( int, int ) );
typedef void    FGAPIENTRY __glutPassiveMotionFunc( void (* callback)( int, int ) );
typedef void    FGAPIENTRY __glutEntryFunc( void (* callback)( int ) );

typedef void    FGAPIENTRY __glutKeyboardUpFunc( void (* callback)( unsigned char, int, int ) );
typedef void    FGAPIENTRY __glutSpecialUpFunc( void (* callback)( int, int, int ) );
typedef void    FGAPIENTRY __glutJoystickFunc( void (* callback)( unsigned int, int, int, int ), int pollInterval );
typedef void    FGAPIENTRY __glutMenuStateFunc( void (* callback)( int ) );
typedef void    FGAPIENTRY __glutMenuStatusFunc( void (* callback)( int, int, int ) );
typedef void    FGAPIENTRY __glutOverlayDisplayFunc( void (* callback)( void ) );
typedef void    FGAPIENTRY __glutWindowStatusFunc( void (* callback)( int ) );

typedef void    FGAPIENTRY __glutSpaceballMotionFunc( void (* callback)( int, int, int ) );
typedef void    FGAPIENTRY __glutSpaceballRotateFunc( void (* callback)( int, int, int ) );
typedef void    FGAPIENTRY __glutSpaceballButtonFunc( void (* callback)( int, int ) );
typedef void    FGAPIENTRY __glutButtonBoxFunc( void (* callback)( int, int ) );
typedef void    FGAPIENTRY __glutDialsFunc( void (* callback)( int, int ) );
typedef void    FGAPIENTRY __glutTabletMotionFunc( void (* callback)( int, int ) );
typedef void    FGAPIENTRY __glutTabletButtonFunc( void (* callback)( int, int, int, int ) );

/*
 * State setting and retrieval functions, see free__glut_state.c
 */
typedef int     FGAPIENTRY __glutGet( GLenum query );
typedef int     FGAPIENTRY __glutDeviceGet( GLenum query );
typedef int     FGAPIENTRY __glutGetModifiers( void );
typedef int     FGAPIENTRY __glutLayerGet( GLenum query );

/*
 * Font stuff, see free__glut_font.c
 */
typedef void    FGAPIENTRY __glutBitmapCharacter( void* font, int character );
typedef int     FGAPIENTRY __glutBitmapWidth( void* font, int character );
typedef void    FGAPIENTRY __glutStrokeCharacter( void* font, int character );
typedef int     FGAPIENTRY __glutStrokeWidth( void* font, int character );
typedef int     FGAPIENTRY __glutBitmapLength( void* font, const unsigned char* string );
typedef int     FGAPIENTRY __glutStrokeLength( void* font, const unsigned char* string );

/*
 * Geometry functions, see free__glut_geometry.c
 */
typedef void    FGAPIENTRY __glutWireCube( GLdouble size );
typedef void    FGAPIENTRY __glutSolidCube( GLdouble size );
typedef void    FGAPIENTRY __glutWireSphere( GLdouble radius, GLint slices, GLint stacks );
typedef void    FGAPIENTRY __glutSolidSphere( GLdouble radius, GLint slices, GLint stacks );
typedef void    FGAPIENTRY __glutWireCone( GLdouble base, GLdouble height, GLint slices, GLint stacks );
typedef void    FGAPIENTRY __glutSolidCone( GLdouble base, GLdouble height, GLint slices, GLint stacks );

typedef void    FGAPIENTRY __glutWireTorus( GLdouble innerRadius, GLdouble outerRadius, GLint sides, GLint rings );
typedef void    FGAPIENTRY __glutSolidTorus( GLdouble innerRadius, GLdouble outerRadius, GLint sides, GLint rings );
typedef void    FGAPIENTRY __glutWireDodecahedron( void );
typedef void    FGAPIENTRY __glutSolidDodecahedron( void );
typedef void    FGAPIENTRY __glutWireOctahedron( void );
typedef void    FGAPIENTRY __glutSolidOctahedron( void );
typedef void    FGAPIENTRY __glutWireTetrahedron( void );
typedef void    FGAPIENTRY __glutSolidTetrahedron( void );
typedef void    FGAPIENTRY __glutWireIcosahedron( void );
typedef void    FGAPIENTRY __glutSolidIcosahedron( void );

/*
 * Teapot rendering functions, found in free__glut_teapot.c
 */
typedef void    FGAPIENTRY __glutWireTeapot( GLdouble size );
typedef void    FGAPIENTRY __glutSolidTeapot( GLdouble size );

/*
 * Game mode functions, see free__glut_gamemode.c
 */
typedef void    FGAPIENTRY __glutGameModeString( const char* string );
typedef int     FGAPIENTRY __glutEnterGameMode( void );
typedef void    FGAPIENTRY __glutLeaveGameMode( void );
typedef int     FGAPIENTRY __glutGameModeGet( GLenum query );

/*
 * Video resize functions, see free__glut_videoresize.c
 */
typedef int     FGAPIENTRY __glutVideoResizeGet( GLenum query );
typedef void    FGAPIENTRY __glutSetupVideoResizing( void );
typedef void    FGAPIENTRY __glutStopVideoResizing( void );
typedef void    FGAPIENTRY __glutVideoResize( int x, int y, int width, int height );
typedef void    FGAPIENTRY __glutVideoPan( int x, int y, int width, int height );

/*
 * Colormap functions, see free__glut_misc.c
 */
typedef void    FGAPIENTRY __glutSetColor( int color, GLfloat red, GLfloat green, GLfloat blue );
typedef GLfloat FGAPIENTRY __glutGetColor( int color, int component );
typedef void    FGAPIENTRY __glutCopyColormap( int window );

/*
 * Misc keyboard and joystick functions, see free__glut_misc.c
 */
typedef void    FGAPIENTRY __glutIgnoreKeyRepeat( int ignore );
typedef void    FGAPIENTRY __glutSetKeyRepeat( int repeatMode );
typedef void    FGAPIENTRY __glutForceJoystickFunc( void );

/*
 * Misc functions, see free__glut_misc.c
 */
typedef int     FGAPIENTRY __glutExtensionSupported( const char* extension );
typedef void    FGAPIENTRY __glutReportErrors( void );

////////////////////////////////////////////////////////////////////////////////////////////////
///\brief free__glut_ext.h functions
///
////////////////////////////////////////////////////////////////////////////////////////////////

/*
 * Window management functions, see free__glut_window.c
 */
typedef void    FGAPIENTRY __glutFullScreenToggle( void );

/*
 * Window-specific callback functions, see free__glut_callbacks.c
 */
typedef void    FGAPIENTRY __glutMouseWheelFunc( void (* callback)( int, int, int, int ) );
typedef void    FGAPIENTRY __glutCloseFunc( void (* callback)( void ) );
typedef void    FGAPIENTRY __glutWMCloseFunc( void (* callback)( void ) );
/* A. Donev: Also a destruction callback for menus */
typedef void    FGAPIENTRY __glutMenuDestroyFunc( void (* callback)( void ) );

/*
 * State setting and retrieval functions, see free__glut_state.c
 */
typedef void    FGAPIENTRY __glutSetOption ( GLenum option_flag, int value );
typedef int *   FGAPIENTRY __glutGetModeValues(GLenum mode, int * size);
/* A.Donev: User-data manipulation */
typedef void*   FGAPIENTRY __glutGetWindowData( void );
typedef void    FGAPIENTRY __glutSetWindowData(void* data);
typedef void*   FGAPIENTRY __glutGetMenuData( void );
typedef void    FGAPIENTRY __glutSetMenuData(void* data);

/*
 * Font stuff, see free__glut_font.c
 */
typedef int     FGAPIENTRY __glutBitmapHeight( void* font );
typedef GLfloat FGAPIENTRY __glutStrokeHeight( void* font );
typedef void    FGAPIENTRY __glutBitmapString( void* font, const unsigned char *string );
typedef void    FGAPIENTRY __glutStrokeString( void* font, const unsigned char *string );

/*
 * Geometry functions, see free__glut_geometry.c
 */
typedef void    FGAPIENTRY __glutWireRhombicDodecahedron( void );
typedef void    FGAPIENTRY __glutSolidRhombicDodecahedron( void );
typedef void    FGAPIENTRY __glutWireSierpinskiSponge ( int num_levels, GLdouble offset[3], GLdouble scale );
typedef void    FGAPIENTRY __glutSolidSierpinskiSponge ( int num_levels, GLdouble offset[3], GLdouble scale );
typedef void    FGAPIENTRY __glutWireCylinder( GLdouble radius, GLdouble height, GLint slices, GLint stacks);
typedef void    FGAPIENTRY __glutSolidCylinder( GLdouble radius, GLdouble height, GLint slices, GLint stacks);

/*
 * Extension functions, see free__glut_ext.c
 */
typedef void (*glutproc)();
typedef glutproc FGAPIENTRY __glutGetProcAddress( const char *procName );


/*
 * Initialization functions, see free__glut_init.c
 */
typedef void    FGAPIENTRY __glutInitContextVersion( int majorVersion, int minorVersion );
typedef void    FGAPIENTRY __glutInitContextFlags( int flags );
typedef void    FGAPIENTRY __glutInitContextProfile( int profile );

/*
 * GLUT API macro definitions -- the display mode definitions
 */
#define  GLUT_CAPTIONLESS                   0x0400
#define  GLUT_BORDERLESS                    0x0800
#define  GLUT_SRGB                          0x1000

/* Comment from glut.h of classic GLUT:

   Win32 has an annoying issue where there are multiple C run-time
   libraries (CRTs).  If the executable is linked with a different CRT
   from the GLUT DLL, the GLUT DLL will not share the same CRT static
   data seen by the executable.  In particular, atexit callbacks registered
   in the executable will not be called if GLUT calls its (different)
   exit routine).  GLUT is typically built with the
   "/MD" option (the CRT with multithreading DLL support), but the Visual
   C++ linker default is "/ML" (the single threaded CRT).

   One workaround to this issue is requiring users to always link with
   the same CRT as GLUT is compiled with.  That requires users supply a
   non-standard option.  GLUT 3.7 has its own built-in workaround where
   the executable's "exit" function pointer is covertly passed to GLUT.
   GLUT then calls the executable's exit function pointer to ensure that
   any "atexit" calls registered by the application are called if GLUT
   needs to exit.

   Note that the __glut*WithExit routines should NEVER be called directly.
   To avoid the atexit workaround, #define GLUT_DISABLE_ATEXIT_HACK. */

/* to get the prototype for exit() */
/*
#include <stdlib.h>

#if defined(_WIN32) && !defined(GLUT_DISABLE_ATEXIT_HACK) && !defined(__WATCOMC__)
typedef void FGAPIENTRY __glutInitWithExit(int *argcp, char **argv, void (__cdecl *exitfunc)(int));
typedef int FGAPIENTRY __glutCreateWindowWithExit(const char *title, void (__cdecl *exitfunc)(int));
typedef int FGAPIENTRY __glutCreateMenuWithExit(void (* func)(int), void (__cdecl *exitfunc)(int));
#ifndef FREEGLUT_BUILDING_LIB
#if defined(__GNUC__)
#define FGUNUSED __attribute__((unused))
#else
#define FGUNUSED
#endif
static void FGAPIENTRY FGUNUSED glutInit_ATEXIT_HACK(int *argcp, char **argv) { __glutInitWithExit(argcp, argv, exit); }
#define glutInit glutInit_ATEXIT_HACK
static int FGAPIENTRY FGUNUSED glutCreateWindow_ATEXIT_HACK(const char *title) { return __glutCreateWindowWithExit(title, exit); }
#define glutCreateWindow glutCreateWindow_ATEXIT_HACK
static int FGAPIENTRY FGUNUSED glutCreateMenu_ATEXIT_HACK(void (* func)(int)) { return __glutCreateMenuWithExit(func, exit); }
#define glutCreateMenu glutCreateMenu_ATEXIT_HACK
#endif
#endif

*/



////////////////////////////////////////////////////////////////////////////////////////////////
///
/// Pointers to all dynamically linked functions
///
////////////////////////////////////////////////////////////////////////////////////////////////


extern __glutMainLoopEvent						*glutMainLoopEvent;
extern __glutLeaveMainLoop						*glutLeaveMainLoop;
extern __glutExit								*glutExit;
extern __glutInit								*glutInit;
extern __glutInitWindowPosition					*glutInitWindowPosition;
extern __glutInitWindowSize						*glutInitWindowSize;
extern __glutInitDisplayMode					*glutInitDisplayMode;
extern __glutInitDisplayString					*glutInitDisplayString;
extern __glutMainLoop							*glutMainLoop;
extern __glutCreateWindow						*glutCreateWindow;
extern __glutCreateSubWindow					*glutCreateSubWindow;
extern __glutDestroyWindow						*glutDestroyWindow;
extern __glutSetWindow							*glutSetWindow;
extern __glutGetWindow							*glutGetWindow;
extern __glutSetWindowTitle						*glutSetWindowTitle;
extern __glutSetIconTitle						*glutSetIconTitle;
extern __glutReshapeWindow						*glutReshapeWindow;
extern __glutPositionWindow						*glutPositionWindow;
extern __glutShowWindow							*glutShowWindow;
extern __glutHideWindow							*glutHideWindow;
extern __glutIconifyWindow						*glutIconifyWindow;
extern __glutPushWindow							*glutPushWindow;
extern __glutPopWindow							*glutPopWindow;
extern __glutFullScreen							*glutFullScreen;
extern __glutPostWindowRedisplay				*glutPostWindowRedisplay;
extern __glutPostRedisplay						*glutPostRedisplay;
extern __glutSwapBuffers						*glutSwapBuffers;
extern __glutWarpPointer						*glutWarpPointer;
extern __glutSetCursor							*glutSetCursor;
extern __glutEstablishOverlay					*glutEstablishOverlay;
extern __glutRemoveOverlay						*glutRemoveOverlay;
extern __glutUseLayer							*glutUseLayer;
extern __glutPostOverlayRedisplay				*glutPostOverlayRedisplay;
extern __glutPostWindowOverlayRedisplay			*glutPostWindowOverlayRedisplay;
extern __glutShowOverlay						*glutShowOverlay;
extern __glutHideOverlay						*glutHideOverlay;
extern __glutCreateMenu							*glutCreateMenu;
extern __glutDestroyMenu						*glutDestroyMenu;
extern __glutGetMenu							*glutGetMenu;
extern __glutSetMenu							*glutSetMenu;
extern __glutAddMenuEntry						*glutAddMenuEntry;
extern __glutAddSubMenu							*glutAddSubMenu;
extern __glutChangeToMenuEntry					*glutChangeToMenuEntry;
extern __glutChangeToSubMenu					*glutChangeToSubMenu;
extern __glutRemoveMenuItem						*glutRemoveMenuItem;
extern __glutAttachMenu							*glutAttachMenu;
extern __glutDetachMenu							*glutDetachMenu;
extern __glutTimerFunc							*glutTimerFunc;
extern __glutIdleFunc							*glutIdleFunc;
extern __glutKeyboardFunc						*glutKeyboardFunc;
extern __glutSpecialFunc						*glutSpecialFunc;
extern __glutReshapeFunc						*glutReshapeFunc;
extern __glutVisibilityFunc						*glutVisibilityFunc;
extern __glutDisplayFunc						*glutDisplayFunc;
extern __glutMouseFunc							*glutMouseFunc;
extern __glutMotionFunc							*glutMotionFunc;
extern __glutPassiveMotionFunc					*glutPassiveMotionFunc;
extern __glutEntryFunc							*glutEntryFunc;
extern __glutKeyboardUpFunc						*glutKeyboardUpFunc;
extern __glutSpecialUpFunc						*glutSpecialUpFunc;
extern __glutJoystickFunc						*glutJoystickFunc;
extern __glutMenuStateFunc						*glutMenuStateFunc;
extern __glutMenuStatusFunc						*glutMenuStatusFunc;
extern __glutOverlayDisplayFunc					*glutOverlayDisplayFunc;
extern __glutWindowStatusFunc					*glutWindowStatusFunc;
extern __glutSpaceballMotionFunc				*glutSpaceballMotionFunc;
extern __glutSpaceballRotateFunc				*glutSpaceballRotateFunc;
extern __glutSpaceballButtonFunc				*glutSpaceballButtonFunc;
extern __glutButtonBoxFunc						*glutButtonBoxFunc;
extern __glutDialsFunc							*glutDialsFunc;
extern __glutTabletMotionFunc					*glutTabletMotionFunc;
extern __glutTabletButtonFunc					*glutTabletButtonFunc;
extern __glutGet								*glutGet;
extern __glutDeviceGet							*glutDeviceGet;
extern __glutGetModifiers						*glutGetModifiers;
extern __glutLayerGet							*glutLayerGet;
extern __glutBitmapCharacter					*glutBitmapCharacter;
extern __glutBitmapWidth						*glutBitmapWidth;
extern __glutStrokeCharacter					*glutStrokeCharacter;
extern __glutStrokeWidth						*glutStrokeWidth;
extern __glutBitmapLength						*glutBitmapLength;
extern __glutStrokeLength						*glutStrokeLength;
extern __glutWireCube							*glutWireCube;
extern __glutSolidCube							*glutSolidCube;
extern __glutWireSphere							*glutWireSphere;
extern __glutSolidSphere						*glutSolidSphere;
extern __glutWireCone							*glutWireCone;
extern __glutSolidCone							*glutSolidCone;
extern __glutWireTorus							*glutWireTorus;
extern __glutSolidTorus							*glutSolidTorus;
extern __glutWireDodecahedron					*glutWireDodecahedron;
extern __glutSolidDodecahedron					*glutSolidDodecahedron;
extern __glutWireOctahedron						*glutWireOctahedron;
extern __glutSolidOctahedron					*glutSolidOctahedron;
extern __glutWireTetrahedron					*glutWireTetrahedron;
extern __glutSolidTetrahedron					*glutSolidTetrahedron;
extern __glutWireIcosahedron					*glutWireIcosahedron;
extern __glutSolidIcosahedron					*glutSolidIcosahedron;
extern __glutWireTeapot							*glutWireTeapot;
extern __glutSolidTeapot						*glutSolidTeapot;
extern __glutGameModeString						*glutGameModeString;
extern __glutEnterGameMode						*glutEnterGameMode;
extern __glutLeaveGameMode						*glutLeaveGameMode;
extern __glutGameModeGet						*glutGameModeGet;
extern __glutVideoResizeGet						*glutVideoResizeGet;
extern __glutSetupVideoResizing					*glutSetupVideoResizing;
extern __glutStopVideoResizing					*glutStopVideoResizing;
extern __glutVideoResize						*glutVideoResize;
extern __glutVideoPan							*glutVideoPan;
extern __glutSetColor							*glutSetColor;
extern __glutGetColor							*glutGetColor;
extern __glutCopyColormap						*glutCopyColormap;
extern __glutIgnoreKeyRepeat					*glutIgnoreKeyRepeat;
extern __glutSetKeyRepeat						*glutSetKeyRepeat;
extern __glutForceJoystickFunc					*glutForceJoystickFunc;
extern __glutExtensionSupported					*glutExtensionSupported;
extern __glutReportErrors						*glutReportErrors;
extern __glutFullScreenToggle					*glutFullScreenToggle;
extern __glutMouseWheelFunc						*glutMouseWheelFunc;
extern __glutCloseFunc							*glutCloseFunc;
extern __glutWMCloseFunc						*glutWMCloseFunc;
extern __glutMenuDestroyFunc					*glutMenuDestroyFunc;
extern __glutSetOption							*glutSetOption;
extern __glutGetModeValues						*glutGetModeValues;
extern __glutGetWindowData						*glutGetWindowData;
extern __glutSetWindowData						*glutSetWindowData;
extern __glutGetMenuData						*glutGetMenuData;
extern __glutSetMenuData						*glutSetMenuData;
extern __glutBitmapHeight						*glutBitmapHeight;
extern __glutStrokeHeight						*glutStrokeHeight;
extern __glutBitmapString						*glutBitmapString;
extern __glutStrokeString						*glutStrokeString;
extern __glutWireRhombicDodecahedron			*glutWireRhombicDodecahedron;
extern __glutSolidRhombicDodecahedron			*glutSolidRhombicDodecahedron;
extern __glutWireSierpinskiSponge				*glutWireSierpinskiSponge;
extern __glutSolidSierpinskiSponge				*glutSolidSierpinskiSponge;
extern __glutWireCylinder						*glutWireCylinder;
extern __glutSolidCylinder						*glutSolidCylinder;
extern __glutGetProcAddress						*glutGetProcAddress;
extern __glutInitContextVersion					*glutInitContextVersion;
extern __glutInitContextFlags					*glutInitContextFlags;
extern __glutInitContextProfile					*glutInitContextProfile;



typedef enum FG_LibLoadCode_enum
{
	FG_SUCCESS,						///< No error
	FG_ERROR_FILE_NOT_FOUND,		///< File not found
	FG_ERROR_SYMBOL_NOT_FOUND,		///< Symbol not found when loading shared library
	FG_ERROR_UNKNOWN				///< Unknown error
}FG_LibLoadCode; 


extern FG_LibLoadCode glutLoadLibrary();

#ifdef __cplusplus
    }
#endif

/*** END OF FILE ***/

#endif /* _FREEGLUT_DYNLINK_H */
