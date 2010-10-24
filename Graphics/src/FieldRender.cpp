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

#include <GL/glew.h>
#include "GL/freeglut.h"
#include "GL/glutExtra.h"
#include "FieldRender class.h"
#include "X-Compat/HPC Timing.h"
#include <stdio.h> // for snprintf()

#if defined(_MSC_VER)
#pragma warning(disable:1786)
#pragma warning(disable:4996)
#endif//MSVC

Camera FieldRender::mainCam;
ProjectionMode FieldRender::PM;
volatile bool FieldRender::quitFlag = false;
FieldRenderer::GLpacket FieldRender::GLdata;
GLuint FieldRender::chargesVBO;
GLuint FieldRender::colorVBO;
GLuint *FieldRender::linesVBOs;
size_t FieldRender::nrLinesVBO;
GLfloat *FieldRender::colors;
size_t FieldRender::lineSkip;
size_t FieldRender::RenderData::bufferedLines;
double FieldRender::perfGFLOP;

static long long HPCfreq;
// DEBUG

Vector2<GLint> winDim = {250, 250};
GLdouble Zmin = 1, Zmax = 10000;

FieldRender::FieldRender()
{
	PM = Perspective;
	lineSkip = 1;
	nrLinesVBO = 0;
	perfGFLOP = 0;
	this->dataBound = false;
}

FieldRender::~FieldRender()
{
	//glDeleteBuffersARB((GLsizei)nrLinesVBO, linesVBOs);
	//__glewDeleteBuffersARB((GLsizei)nrLinesVBO, linesVBOs);
}
void FieldRender::DrawOverlay()//const Camera mainCam)
{
	// draw menus
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluOrtho2D(0, winDim.x, winDim.y, 0);
	
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	// Draw Background
	GLfloat bgHeight = 110, bgWidth = 176;
	glColor4f(0.0f, 0.4f, 0.0f, 0.5f);
	glBegin(GL_POLYGON);
	glVertex3f(0,0,0);
	glVertex3f(bgWidth, 0,0);
	glVertex3f(bgWidth,bgHeight,0);
	glVertex3f(0,bgHeight,0);
	glEnd();
	// Draw the G-Tech logo
	glLoadIdentity();
	glColor3f(1.0, 0.0, 0.0);
	PrintGlutString("G-Tech", GLUT_BITMAP_TIMES_ROMAN_24, 40, 40);
	char fovString[32];
	snprintf(fovString, 31, "FOV: %.0f", mainCam.GetFOV());
	PrintGlutString(fovString, GLUT_BITMAP_HELVETICA_12, 20, 60);
	char camPosStr[32];
	snprintf(camPosStr, 31, "Pos:<%.0f, %.0f, %.0f>", mainCam.GetPosition().x, mainCam.GetPosition().y, mainCam.GetPosition().z);
	PrintGlutString(camPosStr, GLUT_BITMAP_HELVETICA_12, 20, 75);
	char perf[32];
	snprintf(perf, 31, "Perf: %4.0f GFLOP/s", perfGFLOP);
	PrintGlutString(perf, GLUT_BITMAP_HELVETICA_12, 20, 105);

	// Compute and print FPS
	static long long start = 0;
	static long long end = 0;
	static size_t frames = 0;
	static char fps[32];
	QueryHPCTimer(&end);
	double elapsed = (double)(end - start)/HPCfreq;
	if( (elapsed > 1.0) && frames)
	{
		snprintf(fps, 31, "FPS: %.1f", (double)frames/elapsed);
		frames = 0;
		start = end;
	}
	frames++;
	PrintGlutString(fps, GLUT_BITMAP_HELVETICA_12, 20, 90);

}

void FieldRender::fieldDisplay()
{
	glClear(GL_COLOR_BUFFER_BIT);
	glClear(GL_DEPTH_BUFFER_BIT);
	glEnableClientState(GL_VERTEX_ARRAY);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	switch(PM)
	{
	case Perspective:
		gluPerspective(mainCam.GetFOV(), (GLdouble)winDim.x/(GLdouble)winDim.y, Zmin, Zmax);
		break;
	case Orthogonal:
		glOrtho(-winDim.x/2, winDim.x/2, -winDim.y/2, winDim.y/2, Zmin, Zmax);
		break;
	default:
		break;
	}
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	Vector3<GLdouble> camPos = mainCam.GetPosition(),
		camCenter = mainCam.GetCenter(),
		camUp = mainCam.GetUp();
	gluLookAt(camPos.x , camPos.y, camPos.z,
		camCenter.x, camCenter.y, camCenter.z,
		camUp.x, camUp.y, camUp.z);

	// Draw origin axes
	glColor3f(1.0, 1.0, 1.0);
	glBegin(GL_LINES);
	glVertex3f(0,0,0); glVertex3f(100,0,0);// glVertex3f(0,100,0);	//x
	glVertex3f(0,0,0); glVertex3f(0,100,0);// glVertex3f(0,0,100);	//y
	glVertex3f(0,0,0); glVertex3f(0,0,100);// glVertex3f(100,0,0);	//z
	glEnd();

	// Draw Charges
	glColor3f(0.0, 0.0, 1.0);
	glVertexPointer(3 , GL_FLOAT, (GLint)GLdata.charges->GetElemSize(), GLdata.charges->GetDataPointer());
	glDrawArrays(GL_POINTS, 0, (GLint)GLdata.charges->GetSize());

	// Draw lines
	glColor3f(1.0, 0.0, 0.0);
	glEnableClientState(GL_COLOR_ARRAY);
	glColorPointer(3, GL_FLOAT, 0, colors);
	int pts = (GLint)GLdata.lineLen;
	for(size_t i = 0; i<GLdata.nlines; i++)
	{
		glVertexPointer(3 , GL_FLOAT, (GLint)(GLdata.lines->GetElemSize()*GLdata.lines->GetSize()/GLdata.lineLen), GLdata.lines->GetDataPointer()+i);
		glDrawArrays(GL_LINE_STRIP, 0, pts);
		
	}
	glDisableClientState(GL_COLOR_ARRAY);

	// draw menus
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluOrtho2D(0, winDim.x, winDim.y, 0);
	
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	
	glColor4f(0.0f, 0.2f, 0.0f, 0.5f);
	glBegin(GL_POLYGON);
	glVertex3f(0,0,0);
	glVertex3f(176, 0,0);
	glVertex3f(176,99,0);
	glVertex3f(0,99,0);
	glEnd();
	
	// Draw infobar
	DrawOverlay();//mainCam);

	// Flush the buffer to ensure everything is displayed correctly
	glutSwapBuffers();
}

void FieldRender::Start()
{
	// Generate colors
	const size_t elements = GLdata.nlines;
	const size_t elemLen = GLdata.lineLen;
	Vector3<float> *tempBuf = new Vector3<float>[elemLen];
	// Red to green transition
	for(size_t i = 0; i < elemLen/2; i++)
	{
		tempBuf[i].x = (float)(elemLen - 2*i)/elemLen;		// Red chanel
		tempBuf[i].y = (float)(2*i)/elemLen;				// Green channel
		tempBuf[i].z = 0;									// Blue channel
	}
	// Green to blue transition
	for(size_t i = elemLen/2; i < elemLen; i++)
	{
		tempBuf[i].x = 0;											// Red chanel
		tempBuf[i].y = (float)(elemLen- 2*(i-elemLen/2))/elemLen;	// Green channel
		tempBuf[i].z = 2*(float)(i-elemLen/2)/elemLen;				// Blue channel
	}

	colors = (GLfloat*)tempBuf;
	// Now release tempBuf
	tempBuf = 0;

	nrLinesVBO = 0;

	// Enable VBOs if supported and copy data to GPU, and use VBOs for increased performance
	if(VBOsupported)
	{
		// Copy the charges to a VBO
		glGenBuffersARB(1, &chargesVBO);
		glBindBufferARB(GL_ARRAY_BUFFER_ARB, chargesVBO);
		glBufferDataARB(GL_ARRAY_BUFFER_ARB, GLdata.charges->GetSizeBytes(), GLdata.charges->GetDataPointer(), GL_STATIC_DRAW_ARB );

		// Copy the colors to a VBO
		glGenBuffersARB(1, &colorVBO);
		glBindBufferARB(GL_ARRAY_BUFFER_ARB, colorVBO);
		glBufferDataARB(GL_ARRAY_BUFFER_ARB, elemLen * sizeof(Vector3<float>), colors, GL_STATIC_DRAW_ARB );


		// Create VBO index array
		nrLinesVBO = (elements + lineSkip - 1)/lineSkip;
		RenderData::bufferedLines = nrLinesVBO;
		linesVBOs = new GLuint[nrLinesVBO];
		tempBuf = new Vector3<float>[elemLen];
		// Create the buffers
		__glewGenBuffersARB((GLsizei)nrLinesVBO, linesVBOs);
		// can replace __glew* with gl*
		// Copy all the field lines to the GPU in separate arrays
		// Since the field lines comes arranged in lines by steps, the memory arrangement will be n0_0 n1_0 n2_0 n3_0 n4_0... n0_1 n1_1 n2_1 n3_1 n4_1
		// In order to display the data, we need it in n0_0, n0_1, n0_2... form, so we need to copy the data to the temporary buffer.
		// This is easily done by setting the base to ni_0, and for each element adding an offset of GLdata.nlines.
		for(size_t i = 0; i < nrLinesVBO; i++)
		{
			// Set the base to ni_0
			Vector3<float> *temp = GLdata.lines->GetDataPointer() + i*lineSkip;
			// Copy ni_0, ni_1, ni_2... to a linear array
			for(size_t cpy = 0; cpy < elemLen; cpy++)
			{
				tempBuf[cpy] = temp[cpy*elements];
			}
			// Now that the data for ni is copied in linear memory, bind it and copy it to a VBO
			::__glewBindBufferARB(GL_ARRAY_BUFFER_ARB, linesVBOs[i]);
			::__glewBufferDataARB(GL_ARRAY_BUFFER_ARB,
				GLdata.lines->GetElemSize()*elemLen,
				tempBuf, GL_STATIC_DRAW_ARB );
		}
		// Unbind the buffers from any specific VBO
		glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);
		// Clean up temporary resources
		delete[] tempBuf;
		// We also don't need the color buffer on the client
		delete[] colors;
		colors = 0;
		// Finally, override the regular display function to use VBOs
		glutDisplayFunc(fieldDisplayVBO);
	}
	else fprintf(stderr, "Warning: GL_ARB_vertex_buffer_object extension not supported. Rendering may be painfully slow!!!\n");
	// Get timer frequency
	QueryHPCFrequency(&HPCfreq);
	// Now that data initialization is complete, start glut
	glutMainLoop();
}

void FieldRender::GLInit()
{
    // Display initialization
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
		
	// GLUT window management initialization
	glutCreateWindow ("Electromag - Electrostatics mode");
	glutFullScreen();
    
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	// Antialiasing gives the field lines a very grainy appearance
	//glEnable(GL_LINE_SMOOTH);
	//glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);

	// GLUT input handler functions
	glutDisplayFunc(fieldDisplay);
	glutReshapeFunc(reshape);
	glutKeyboardFunc(keyboard);
	glutMouseFunc(mouse);		
	glutMotionFunc(motion);
	glutIdleFunc((void (*)(void))glutPostRedisplay);
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	glClearDepth(1.0f);

	//----------------------------------Check for VBO support----------------------------------//
	// start GLEW
	glewInit();
	VBOsupported = glewIsSupported("GL_ARB_vertex_buffer_object");		
}

void FieldRender::reshape(int w, int h)
{
	winDim.x = w; winDim.y = h;
	glViewport(0, 0 , w, h);
}

void FieldRender::keyboard(unsigned char key, int x, int y)
{
	switch(key)
	{
	// Motion
	case 'w':	// Sideways
	case 'W':
		mainCam.Move(10, 0, 0);
		break;
	case 's':
	case 'S':
		mainCam.Move(-10, 0, 0);
		break;
	case 'd':	// Lateral
	case 'D':
		mainCam.Move(0, 10, 0);
		break;
	case 'a':
	case 'A':
		mainCam.Move(0, -10, 0);
		break;
	case 'y':	// Vertical
	case 'Y':
		mainCam.Move(0, 0, 10);
		break;
	case 'h':
	case 'H':
		mainCam.Move(0, 0, -10);
		break;
	// Rotation
	case 'e':	// Lateral
	case 'E':
		mainCam.Rotate(5, 0, Degree);
		break;
	case 'q':
	case 'Q':
		mainCam.Rotate(-5, 0, Degree);
		break;
	case 't':	// Vertical
	case 'T':
		mainCam.Rotate(0, 5, Degree);
		break;
	case 'g':
	case 'G':
		mainCam.Rotate(0, -5, Degree);
		break;
	// Mode change
	case 'm':
	case 'M':
		if(PM == Perspective) PM = Orthogonal;
		else PM = Perspective;
		break;
	// Zoom
	case 'i':
	case 'I':
		mainCam.ZoomLinear(1);
		break;
	case 'o':
	case 'O':
		mainCam.ZoomLinear(-1);
		break;
	case 'r':
	case 'R':
		mainCam.ResetFOV();
		break;
	case 'v':
	case 'V':
		mainCam.ResetPosition();
		break;
	case 'f':
	case 'F':
		// Re-initialize fullscreen mode
		glutFullScreenToggle();
		break;
	// Quick exit method
	case'\033':
		quitFlag = true;
		// Wait for main program to kill renderer
		while(1);
		break;
	}
	//glutPostRedisplay();
}

static int leftButtonState = GLUT_UP;
static int rightButtonState = GLUT_UP;
static Vector2<int> oldMousePos;

void FieldRender::motion(int x, int y)
{
	const int dx = x - oldMousePos.x;
	const int dy = y - oldMousePos.y;
	oldMousePos.x = x; oldMousePos.y = y;
	if(leftButtonState == GLUT_DOWN)
	{
		GLdouble rotY = mainCam.GetFOV() * (GLdouble)dy/winDim.y;
		GLdouble rotX = mainCam.GetFOV() * (GLdouble)winDim.x/winDim.y * (GLdouble)-dx/winDim.x;
		mainCam.Rotate(rotX, rotY, Degree);
	}
	else if(rightButtonState == GLUT_DOWN)
	{
		// A horizontal mose sweep across the whole window should rotate by 360deg or 2*PI
		GLdouble rotX = 2 * PI * (GLdouble)dx/winDim.x;
		GLdouble rotY = 2 * PI * (GLdouble)dy/winDim.y * (GLdouble)winDim.y/winDim.x ;
		mainCam.RotateAroundCenter(rotX, rotY, Radian);
	}
}

#define WHEEL_UP 3
#define WHEEL_DOWN 4
void FieldRender::mouse(int button, int state, int x, int y)
{
	// Keep track of the mouse position. this will be needed by the motion function
	oldMousePos.x = x;
	oldMousePos.y = y;

	
	switch(button)
	{
	case GLUT_LEFT_BUTTON:
		leftButtonState = state;
		break;
	case GLUT_RIGHT_BUTTON:
		rightButtonState = state;
		break;
	case GLUT_MIDDLE_BUTTON:
		break;
	case WHEEL_UP:
		if(state == GLUT_DOWN) mainCam.ZoomExponential(10);
		break;
	case WHEEL_DOWN:
		if(state == GLUT_DOWN) mainCam.ZoomExponential(-10);
		break;
	default:
		break;
	}
}

void FieldRender::fieldDisplayVBO()
{
	// Clear any buffers
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glEnableClientState(GL_VERTEX_ARRAY);
	
	// Select proper projection mode
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	switch(PM)
	{
	case Perspective:
		gluPerspective(mainCam.GetFOV(), (GLdouble)winDim.x/(GLdouble)winDim.y, Zmin, Zmax);
		break;
	case Orthogonal:
		glOrtho(-winDim.x/2, winDim.x/2, -winDim.y/2, winDim.y/2, Zmin, Zmax);
		break;
	default:
		break;
	}
	// Position viewpoint
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	Vector3<GLdouble> camPos = mainCam.GetPosition(),
		camCenter = mainCam.GetCenter(),
		camUp = mainCam.GetUp();
	gluLookAt(camPos.x , camPos.y, camPos.z,
		camCenter.x, camCenter.y, camCenter.z,
		camUp.x, camUp.y, camUp.z);

	// Draw origin axes
	glColor3f(1.0, 1.0, 1.0);
	glBegin(GL_LINES);
	glVertex3f(0,0,0); glVertex3f(100,0,0);	//x
	glVertex3f(0,0,0); glVertex3f(0,100,0);	//y
	glVertex3f(0,0,0); glVertex3f(0,0,100);	//z
	glEnd();

	// Draw Charges
	glColor3f(0.0, 0.0, 1.0);
	glBindBufferARB( GL_ARRAY_BUFFER_ARB, chargesVBO);
	glVertexPointer(3 , GL_FLOAT, (GLint)GLdata.charges->GetElemSize(),0);
	glDrawArrays(GL_POINTS, 0, (GLint)GLdata.charges->GetSize());


	// Draw field lines
	glEnableClientState(GL_COLOR_ARRAY);
	glBindBufferARB(GL_ARRAY_BUFFER_ARB, colorVBO);
	glColorPointer(3, GL_FLOAT, 0, 0);
	glColor3f(1.0, 0.0, 0.0);
	for(size_t i = 0; i < RenderData::bufferedLines; i++)
	{
		glBindBufferARB(GL_ARRAY_BUFFER_ARB, linesVBOs[i]);
		glVertexPointer(3 , GL_FLOAT, 0, 0);
		glDrawArrays(GL_LINE_STRIP, 0, (GLsizei)GLdata.lineLen);
	}
	glDisableClientState(GL_COLOR_ARRAY);
	glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);
	

	// Draw infobar in top-right corner
	DrawOverlay();

	// Flush the buffer to ensure everything is displayed correctly
	glutSwapBuffers();
}

void FieldRender::AsyncStartFunc()
{
	if(!this->dataBound) return;
	try
	{
		Init();
	}
	catch(char *errString)
	{
		fprintf(stderr, " Initialing field rendering failed.\n %s\n", errString);
		this->quitFlag = true;
		return;
	}
	SetLineSkip(20);
	Start();
}
