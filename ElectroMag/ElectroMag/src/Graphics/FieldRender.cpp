/*////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
GL/glew.h and GL/glut.h are found in the nVidia CUDA SDK
For Windows, must link to both freeglut.lib, and glew64.lib, and have freeglut.dll and glew64.dll in the application path
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////*/
#include <GL/glew.h>
#include "GL/freeglut.h"
#include "GL/glutExtra.h"
#include "FieldRender.h"
#include "./../X-Compat/HPC timing.h"
#include <stdio.h> // for sprintf()

#pragma warning(disable:1786)
#pragma warning(disable:4996)

Camera FieldRender::mainCam;
ProjectionMode FieldRender::PM;
volatile bool shouldIQuit = false;
GLpacket<float> FieldRender::GLdata;
GLuint FieldRender::chargesVBO;
GLuint FieldRender::colorVBO;
GLuint *FieldRender::linesVBOs;
GLfloat *FieldRender::colors;
size_t FieldRender::lineSkip;
size_t FieldRender::RenderData::bufferedLines;

static __int64 HPCfreq;
// DEBUG

Vector2<GLint> winDim = {250, 250};
GLdouble Zmin = 1, Zmax = 10000;


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
	for(int i = 0; i<GLdata.nlines; i++)
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
	
	glColor4f(0.0, 0.2, 0.0, 0.5);
	glBegin(GL_POLYGON);
	glVertex3f(0,0,0);
	glVertex3f(176, 0,0);
	glVertex3f(176,99,0);
	glVertex3f(0,99,0);
	glEnd();
	glLoadIdentity();
	glColor3f(1.0, 0.0, 0.0);
	PrintGlutString("G-Tech", GLUT_BITMAP_TIMES_ROMAN_24, 40, 40);

	// Flush the buffer to ensure everything is displayed correctly
	glutSwapBuffers();
};

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
		const size_t dispElems = (elements + lineSkip - 1)/lineSkip;
		RenderData::bufferedLines = dispElems;
		linesVBOs = new GLuint[dispElems];
		tempBuf = new Vector3<float>[elemLen];
		// Create the buffers
		glGenBuffersARB(dispElems, linesVBOs);			
		// can replace __glew* with gl*
		// Copy all the field lines to the GPU in separate arrays
		// Since the field lines comes arranged in lines by steps, the memory arrangement will be n0_0 n1_0 n2_0 n3_0 n4_0... n0_1 n1_1 n2_1 n3_1 n4_1
		// In order to display the data, we need it in n0_0, n0_1, n0_2... form, so we need to copy the data to the temporary buffer.
		// This is easily done by setting the base to ni_0, and for each element adding an offset of GLdata.nlines.
		for(int i = 0; i < dispElems; i++)
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
	else fprintf(stderr, "Warning: Unable to use VBO extensions. Rendering may be painfully slow!!!\n");
	// Get timer frequency
	QueryHPCFrequency(&HPCfreq);
	// Now that data initialization is complete, start glut
	glutMainLoop();
};

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
};

void FieldRender::reshape(int w, int h)
{
	winDim.x = w; winDim.y = h;
	glViewport(0, 0 , w, h);
};

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
		shouldIQuit = true;
		// Wait for main program to kill renderer
		while(1);
		break;
	}
	//glutPostRedisplay();
}

static int leftButtonState = GLUT_UP;
static Vector2<int> oldMousePos;

void FieldRender::motion(int x, int y)
{
	if(leftButtonState == GLUT_UP) return;
	const int dx = oldMousePos.x - x;
	const int dy = y - oldMousePos.y;
	oldMousePos.x = x; oldMousePos.y = y;
	GLdouble rotY = mainCam.GetFOV() * (GLdouble)dy/winDim.y;
	GLdouble rotX = mainCam.GetFOV() * (GLdouble)winDim.x/winDim.y * (GLdouble)dx/winDim.x;
	mainCam.Rotate(rotX, rotY, Degree);
	//glutPostRedisplay();
}

#define WHEEL_UP 3
#define WHEEL_DOWN 4
void FieldRender::mouse(int button, int state, int x, int y)
{
	if (button == GLUT_LEFT_BUTTON) leftButtonState = state;
	oldMousePos.x = x;
	oldMousePos.y = y;

	if(state == GLUT_DOWN)
	switch(button)
	{
	case GLUT_LEFT_BUTTON:
		break;
	case GLUT_RIGHT_BUTTON:
		break;
	case GLUT_MIDDLE_BUTTON:
		break;
	case WHEEL_UP:
		mainCam.ZoomExponential(10);
		break;
	case WHEEL_DOWN:
		mainCam.ZoomExponential(-10);
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
	for(int i = 0; i < RenderData::bufferedLines; i++)
	{
		glBindBufferARB(GL_ARRAY_BUFFER_ARB, linesVBOs[i]);
		glVertexPointer(3 , GL_FLOAT, 0, 0);
		glDrawArrays(GL_LINE_STRIP, 0, GLdata.lineLen);
	}
	glDisableClientState(GL_COLOR_ARRAY);
	glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);

	// draw menus
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluOrtho2D(0, winDim.x, winDim.y, 0);
	
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	
	// Draw the G-Tech logo
	glColor4f(0.0, 0.4, 0.0, 0.5);
	glBegin(GL_POLYGON);
	glVertex3f(0,0,0);
	glVertex3f(176, 0,0);
	glVertex3f(176,99,0);
	glVertex3f(0,99,0);
	glEnd();
	glLoadIdentity();
	glColor3f(1.0, 0.0, 0.0);
	PrintGlutString("G-Tech", GLUT_BITMAP_TIMES_ROMAN_24, 40, 40);
	char fovString[20];
	sprintf(fovString, "FOV: %.0f", mainCam.GetFOV());
	PrintGlutString(fovString, GLUT_BITMAP_HELVETICA_12, 20, 60);
	char camPosStr[32];
	sprintf(camPosStr, "Pos:<%.0f, %.0f, %.0f>", mainCam.GetPosition().x, mainCam.GetPosition().y, mainCam.GetPosition().z);
	PrintGlutString(camPosStr, GLUT_BITMAP_HELVETICA_12, 20, 75);

	// Compute and print FPS
	static __int64 start = 0;
	static __int64 end = 0;
	static size_t frames = 0;
	static char fps[32];
	QueryHPCTimer(&end);
	double elapsed = (double)(end - start)/HPCfreq;
	if( (elapsed > 1.0) && frames)
	{
		sprintf(fps, "FPS: %.1f", (double)frames/elapsed);
		frames = 0;
		start = end;
	}
	frames++;
	PrintGlutString(fps, GLUT_BITMAP_HELVETICA_12, 20, 90);

	// Flush the buffer to ensure everything is displayed correctly
	glutSwapBuffers();
};

void FieldRender::AsyncStartFunc()
{
	Init();
	SetLineSkip(20);
	Start();
}
