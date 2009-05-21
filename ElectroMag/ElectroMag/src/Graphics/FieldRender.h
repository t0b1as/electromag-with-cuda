#pragma once
#include "Renderer.h"
#include "Camera.h"
#include "./../Data Structures.h"
#include "./../Electrostatics.h"
#include <stdio.h>
// DEBUG
template<class T>
struct GLpacket
{
	Array<pointCharge<T> > *charges;
	Array<Vector3<T> > *lines;
	size_t nlines, lineLen;
};

enum ProjectionMode {Orthogonal, Perspective};
extern volatile bool shouldIQuit;
 


class FieldRender: public GLRenderer
{
private:
	static GLpacket<float> GLdata;
	static ProjectionMode PM;
	static Camera mainCam;
	static size_t lineSkip;
	void GLInit();
	/*static*/ struct RenderData
	{
		static size_t bufferedLines;
	};

	// GL specific data
	bool VBOsupported;
	static unsigned int chargesVBO, colorVBO;
	static unsigned int *linesVBOs;
	static float* colors;
    
    void AsyncStartFunc();

public:
	FieldRender()
	{
		PM = Perspective;
		lineSkip = 1;
	};
	
	void Draw()
	{
	};

	void RenderPacket(GLpacket<float> data)
	{
		GLdata = data;
	}

	void SetLineSkip(size_t skip)
	{
		lineSkip = skip?skip:1;
	}

	void Start();

private:
    
    unsigned long AsyncStartFunc(void * parameters);
    
	static void fieldDisplay();
	
	static void fieldDisplayVBO();

	static void reshape(int w, int h);

	static void keyboard(unsigned char key, int x, int y);

	static void mouse(int button, int state, int x, int y);

	static void motion(int x, int y);

};

static FieldRender FieldDisp;

