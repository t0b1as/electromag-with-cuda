/***********************************************************************************************
	Electromag - Electomagnestism simulation application using CUDA accelerated computing
	Copyright (C) 2009 - Alexandru Gagniuc - <http:\\g-tech.homeserver.com\HPC.htm>
See Electromag.cpp for license
***********************************************************************************************/
#include "FrontendGUI.h"
#include "../X-Compat/Threading.h"



FrontendGUI::FrontendGUI()
{
}

FrontendGUI::~FrontendGUI()
{
}
    
void FrontendGUI::Draw()
{
}

void FrontendGUI::frontendDisplay()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
    glColor3f(1.0, 1.0, 1.0);
    PrintGlutString("Experimental Frontend for Electromag",GLUT_BITMAP_HELVETICA_12, 40, 40, 0);
    glutSwapBuffers();
}

void FrontendGUI::GLInit()
{
    // Display initialization
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
		
	// GLUT window management initialization
	glutCreateWindow ("Electromag");
    ::glutInitWindowPosition(100,100);
    ::glutInitWindowSize(600, 400);
	//glutFullScreen();

	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	// Antialiasing gives the field lines a very grainy appearance
	//glEnable(GL_LINE_SMOOTH);
	//glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);

	// GLUT input handler functions
	glutDisplayFunc(frontendDisplay);
	glutReshapeFunc(reshape);
	glutKeyboardFunc(keyboard);
	glutMouseFunc(mouse);		
	glutMotionFunc(motion);
	glutIdleFunc((void (*)(void))glutPostRedisplay);
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	glClearDepth(1.0f);

}
    
void FrontendGUI::AsyncStartFunc()
{
    Init();
    Start();
}
    
void FrontendGUI::Start()
{
    glutMainLoop();
}
	

void FrontendGUI::reshape(int w, int h)
{
	glViewport(0, 0 , w, h);
    glOrtho(0, 0 , w, h, - 10, 10);
};

void FrontendGUI::keyboard(unsigned char key, int x, int y)
{
    
}

void FrontendGUI::mouse(int button, int state, int x, int y)
{
    
}

void FrontendGUI::motion(int x, int y)
{
    
}