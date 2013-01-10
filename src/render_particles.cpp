/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#include <GL/glew.h>

#include <iostream>
#include <math.h>
#include <assert.h>
#include <stdio.h>

#include <cutil_inline.h>

#include "render_particles.h"
#include "shaders_spheres.h"
#include "shaders_cosmos.h"

#ifndef M_PI
#define M_PI    3.1415926535897932384626433832795
#endif

ParticleRenderer::ParticleRenderer(uint3 dim)
: m_pos(0),
  m_numParticles(0),
  m_particleRadius(0.125f * 0.05f),
  m_program_sphere(0),
  m_program_cosmos(0), 
  m_texture(0),
  m_spriteSize(2.0f),
  m_vbo(0),
  m_colorVBO(0),
  b_renderSkyBox(true),
  b_renderFloor(false),
  floorTex(0),
  _displayMode(PARTICLE_SPHERES)
{
	m_gridDim = dim;
    _initGL();
}

ParticleRenderer::~ParticleRenderer()
{
    m_pos = 0;
    
    if (floorTex)  glDeleteTextures(1, &floorTex);
    if (skyboxTex) { for (int i=0; i<6; ++i) glDeleteTextures(1, &skyboxTex[i]); }
}

void ParticleRenderer::setPositions(float *pos, const int numParticles)
{
    m_pos = pos;
    m_numParticles = numParticles;
}

void ParticleRenderer::setVertexBuffer(unsigned int vbo, int numParticles)
{
    m_vbo = vbo;
    m_numParticles = numParticles;
}

void ParticleRenderer::toggleSkyBox(bool boolean){
	b_renderSkyBox = boolean;
}

void ParticleRenderer::toggleSkyBox(){
	b_renderSkyBox = !b_renderSkyBox;
	b_renderFloor = !b_renderSkyBox;
}

void ParticleRenderer::toggleFloor(bool boolean){
	b_renderFloor = boolean;
}

void ParticleRenderer::toggleFloor(){
	b_renderFloor = !b_renderFloor;
}

void ParticleRenderer::drawSkyBox(){

	glEnable( GL_TEXTURE_2D );

	glColor3f(1.0, 1.0, 1.0);
	glNormal3f(0.0, 1.0, 0.0);

	float x = (float)m_gridDim.x;
	float y = (float)m_gridDim.y;
	float z = (float)m_gridDim.z;
	float rep = 1.f;  // <rep> per width

	glBindTexture( GL_TEXTURE_2D, skyboxTex[0] ); // back (TR/ac)
    glBegin(GL_QUADS);
    {    	
    	glTexCoord2f(0.f, 0.f); glVertex3f(x, 0, z); 
        glTexCoord2f(rep, 0.f); glVertex3f(0, 0, z);  
        glTexCoord2f(rep, rep); glVertex3f(0, 0, 0);  
        glTexCoord2f(0.f, rep); glVertex3f(x, 0, 0);               
	}
	glEnd();
	
	glBindTexture( GL_TEXTURE_2D, skyboxTex[1] ); // down (TL/c)
    glBegin(GL_QUADS);
    {    	
        glTexCoord2f(0.f, 0.f); glVertex3f(0, y, 0);  
        glTexCoord2f(rep, 0.f); glVertex3f(x, y, 0); 
        glTexCoord2f(rep, rep); glVertex3f(x, 0, 0); 
        glTexCoord2f(0.f, rep); glVertex3f(0, 0, 0);         
	}
	glEnd();
	
	glBindTexture( GL_TEXTURE_2D, skyboxTex[2] ); // front (TL/c)
    glBegin(GL_QUADS);
    {    
        // top left (L)
        glTexCoord2f(0.f, 0.f); glVertex3f(0, y, z);  
        glTexCoord2f(rep, 0.f); glVertex3f(x, y, x); 
        glTexCoord2f(rep, rep); glVertex3f(x, y, 0); 
        glTexCoord2f(0.f, rep); glVertex3f(0, y, 0); 
	}
	glEnd();	

	glBindTexture( GL_TEXTURE_2D, skyboxTex[3] ); // left (TL/c) 
    glBegin(GL_QUADS);
    {    	
        glTexCoord2f(0.f, 0.f); glVertex3f(0, 0, z);  
        glTexCoord2f(rep, 0.f); glVertex3f(0, y, z); 
        glTexCoord2f(rep, rep); glVertex3f(0, y, 0); 
        glTexCoord2f(0.f, rep); glVertex3f(0, 0, 0); 
	}
	glEnd();
	
	glBindTexture( GL_TEXTURE_2D, skyboxTex[4] ); // right (TR/ac)
    glBegin(GL_QUADS);
    {    	        
        glTexCoord2f(0.f, 0.f); glVertex3f(x, y, z);  
        glTexCoord2f(rep, 0.f); glVertex3f(x, 0, z); 
        glTexCoord2f(rep, rep); glVertex3f(x, 0, 0); 
        glTexCoord2f(0.f, rep); glVertex3f(x, y, 0);        
	}
	glEnd();
	
	glBindTexture( GL_TEXTURE_2D, skyboxTex[5] ); // up (TR/ac)
    glBegin(GL_QUADS);
    {    	
        glTexCoord2f(0.f, 0.f); glVertex3f(x, y, z);  
        glTexCoord2f(rep, 0.f); glVertex3f(0, y, z); 
        glTexCoord2f(rep, rep); glVertex3f(0, 0, z); 
        glTexCoord2f(0.f, rep); glVertex3f(x, 0, z); 
	}
	glEnd();

	glBindTexture( GL_TEXTURE_2D, 0);
	glDisable( GL_TEXTURE_2D );	
}

/*
void ParticleRenderer::drawSky(){

	glEnable( GL_TEXTURE_2D );
	glBindTexture( GL_TEXTURE_2D, skyTex );
        
	glColor3f(1.0, 1.0, 1.0);
	glNormal3f(0.0, 1.0, 0.0);
	
	float x = (float)m_gridDim.x;
	float y = (float)m_gridDim.y;
	float z = (float)m_gridDim.z;
  	float rep = 1.f;  // <rep> per width
	
    glBegin(GL_QUADS);
    {
    	
        glTexCoord2f(0.f, 0.f); glVertex3f(x, 0, z);  
        glTexCoord2f(rep, 0.f); glVertex3f(0, 0, z); 
        glTexCoord2f(rep, rep); glVertex3f(0, 0, 0); 
        glTexCoord2f(0.f, rep); glVertex3f(x, 0, 0); 
        
        glTexCoord2f(0.f, 0.f); glVertex3f(x, y, z);  
        glTexCoord2f(rep, 0.f); glVertex3f(0, y, z); 
        glTexCoord2f(rep, rep); glVertex3f(0, y, 0); 
        glTexCoord2f(0.f, rep); glVertex3f(x, y, 0); 
        
        glTexCoord2f(0.f, 0.f); glVertex3f(0, 0, z);  
        glTexCoord2f(rep, 0.f); glVertex3f(0, y, z); 
        glTexCoord2f(rep, rep); glVertex3f(0, y, 0); 
        glTexCoord2f(0.f, rep); glVertex3f(0, 0, 0); 
        
        glTexCoord2f(0.f, 0.f); glVertex3f(x, 0, z);  
        glTexCoord2f(rep, 0.f); glVertex3f(x, y, z); 
        glTexCoord2f(rep, rep); glVertex3f(x, y, 0); 
        glTexCoord2f(0.f, rep); glVertex3f(x, 0, 0); 
    }
	glEnd();
	
	glBindTexture( GL_TEXTURE_2D, 0);
	glDisable( GL_TEXTURE_2D );
}
*/

void ParticleRenderer::drawFloor(){

	glEnable( GL_TEXTURE_2D );
	glBindTexture( GL_TEXTURE_2D, floorTex );
        
	glColor3f(1.0, 1.0, 1.0);
	glNormal3f(0.0, 1.0, 0.0);
	
	float x = (float)m_gridDim.x;
	float y = (float)m_gridDim.y;
	float rep = 10.f;
	
    glBegin(GL_QUADS);
    {    	  	  	
        glTexCoord2f(0.f, 0.f); glVertex3f(0, 0, 0); 
        glTexCoord2f(rep, 0.f); glVertex3f(x, 0, 0); 
        glTexCoord2f(rep, rep); glVertex3f(x, y, 0); 
        glTexCoord2f(0.f, rep); glVertex3f(0, y, 0); 
    }
	glEnd();
	
	glBindTexture( GL_TEXTURE_2D, 0);
	glDisable( GL_TEXTURE_2D );
}



void ParticleRenderer::_drawPoints(bool useColorVBO = true)
{
    if (!m_vbo) {
        glBegin(GL_POINTS);
        {
            int k = 0;
            for (int i = 0; i < m_numParticles; ++i)
            {
                glVertex3fv(&m_pos[k]);
                k += 3;
            }
        }
        glEnd();
    } else {    	
        glBindBufferARB(GL_ARRAY_BUFFER_ARB, m_vbo);
        glVertexPointer(3, GL_FLOAT, 0, 0);
        glEnableClientState(GL_VERTEX_ARRAY);                

        if (useColorVBO && m_colorVBO) {
            glBindBufferARB(GL_ARRAY_BUFFER_ARB, m_colorVBO);
            glColorPointer(4, GL_FLOAT, 0, 0);
            glEnableClientState(GL_COLOR_ARRAY);
        }

        glDrawArrays(GL_POINTS, 0, m_numParticles);

        glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);
        glDisableClientState(GL_VERTEX_ARRAY); 
        glDisableClientState(GL_COLOR_ARRAY);        
    }
}

void ParticleRenderer::setNextDisplayMode(){
	_displayMode = (ParticleRenderer::DisplayMode)((_displayMode + 1) % PARTICLE_NUM_MODES);
}

void ParticleRenderer::setDisplayMode(DisplayMode mode){
	_displayMode = mode;
}

void ParticleRenderer::display()
{	
	if (b_renderSkyBox){
		drawSkyBox();
	} 
	if (b_renderFloor){
		drawFloor();
	}
	
    switch (_displayMode)
    { 
		case PARTICLE_POINTS:
		{
		    glColor3f(1, 1, 1);
		    glPointSize(1.0f);
		    _drawPoints();
		}
        break;
		case PARTICLE_FIRE:
		{
			glPointSize(4.0f);
			float m_baseColor[4] = { 0.6f, 0.1f, 0.0f, 0.95f};
			glColor4f(m_baseColor[0],m_baseColor[1],m_baseColor[2],m_baseColor[3]);
			
			glEnable (GL_POINT_SMOOTH);
			glHint (GL_POINT_SMOOTH_HINT, GL_NICEST);
			glAlphaFunc(GL_GREATER, 0.1);
			glEnable(GL_ALPHA_TEST);  	
			glBlendFunc(GL_SRC_ALPHA, GL_ONE);
			glEnable(GL_BLEND);	
			glDepthMask(GL_FALSE); // Disable depth buffer updating! (Integral to transparency!)
			_drawPoints(false);
			glDepthMask(GL_TRUE);
			glDisable(GL_BLEND);
			glDisable(GL_ALPHA_TEST);
		}
		break;
		case SPRITE_COSMOS:
		{			
		    glEnable(GL_POINT_SPRITE_ARB);	// setup point sprites		    
		    glTexEnvi(GL_POINT_SPRITE_ARB, GL_COORD_REPLACE_ARB, GL_TRUE);
		    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE_NV);
		    glBlendFunc(GL_SRC_ALPHA, GL_ONE);
		    glEnable(GL_BLEND);
		    glDepthMask(GL_FALSE);
		    glUseProgram(m_program_cosmos);
		    
		    GLuint texLoc = glGetUniformLocation(m_program_cosmos, "splatTexture");
		    
		    glUniform1i(texLoc, 0);
		    
		    glActiveTextureARB(GL_TEXTURE0_ARB);
		    
		    glBindTexture(GL_TEXTURE_2D, m_texture);
		    glColor3f(1, 1, 1);
		    glPointSize(m_particleRadius*2.0f);
		    //float m_baseColor[4] = {0.6f, 0.1f, 0.0f, 0.95f};
		    float m_baseColor[4] = { 1.0f, 0.6f, 0.3f, 0.20f};	// nbody fp32 color (yellow)
		    //float m_baseColor[4] = { 0.4f, 0.8f, 0.1f, 1.0f}; // nbody fp64 color (green)
		    glSecondaryColor3fv(m_baseColor);        
		    _drawPoints();        
		    glUseProgram(0);
		    glDisable(GL_POINT_SPRITE_ARB);
		    glDisable(GL_BLEND);
		    glDepthMask(GL_TRUE);
		  }    	
		  break;
		default: 
		case PARTICLE_SPHERES:
		{		
		    glEnable(GL_POINT_SPRITE_ARB);
		    glTexEnvi(GL_POINT_SPRITE_ARB, GL_COORD_REPLACE_ARB, GL_TRUE);
		    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE_NV);
		    glDepthMask(GL_TRUE);
		    glEnable(GL_DEPTH_TEST);
		    glUseProgram(m_program_sphere);
		    glUniform1f( glGetUniformLocation(m_program_sphere, "pointScale"), m_window_h / tanf(m_fov*0.5f*(float)M_PI/180.0f) );
		    glUniform1f( glGetUniformLocation(m_program_sphere, "pointRadius"), m_particleRadius );
		    glColor3f(1, 1, 1);
		    _drawPoints();
		    glUseProgram(0);
		    glDisable(GL_POINT_SPRITE_ARB);
		}
		break;
    }
}

GLuint
ParticleRenderer::_compileProgram(const char *vsource, const char *fsource)
{
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);

    glShaderSource(vertexShader, 1, &vsource, 0);
    glShaderSource(fragmentShader, 1, &fsource, 0);
    
    glCompileShader(vertexShader);
    glCompileShader(fragmentShader);

    GLuint program = glCreateProgram();

    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);

    glLinkProgram(program);

    // check if program linked
    GLint success = 0;
    glGetProgramiv(program, GL_LINK_STATUS, &success);

    if (!success) {
        char temp[256];
        glGetProgramInfoLog(program, 256, 0, temp);
        printf("Failed to link program:\n%s\n", temp);
        glDeleteProgram(program);
        program = 0;
    }

    return program;
}

void ParticleRenderer::_initGL()
{
    m_program_sphere = _compileProgram(vertexShader_sphere, pixelShader_sphere);
	m_program_cosmos   = _compileProgram(vertexShader_cosmos,   pixelShader_cosmos); _createTexture(32);

#if !defined(__APPLE__) && !defined(MACOSX)
    glClampColorARB(GL_CLAMP_VERTEX_COLOR_ARB, GL_FALSE);
    glClampColorARB(GL_CLAMP_FRAGMENT_COLOR_ARB, GL_FALSE);
#endif

	// load floor texture
	char imagePath[] = "../src/data/floortile.ppm";
    if (imagePath == 0) {
        fprintf(stderr, "Error finding floor image file\n");
        fprintf(stderr, "  FAILED\n");
        exit(EXIT_FAILURE);
    }
    floorTex = loadTexture(imagePath);
   
    // load sky texture
	char imagePath2[] = "../src/data/pansky2.ppm";
    if (imagePath2 == 0) {
        fprintf(stderr, "Error finding floor image file\n");
        fprintf(stderr, "  FAILED\n");
        exit(EXIT_FAILURE);
    }
    
    skyboxTex[0] = loadTexture("../src/data/faesky02back.ppm");
    skyboxTex[1] = loadTexture("../src/data/faesky02down.ppm");
    skyboxTex[2] = loadTexture("../src/data/faesky02front.ppm");
    skyboxTex[3] = loadTexture("../src/data/faesky02left.ppm");
    skyboxTex[4] = loadTexture("../src/data/faesky02right.ppm");
    skyboxTex[5] = loadTexture("../src/data/faesky02up.ppm");    
     
    for (int i=0; i<6; i++){
		glBindTexture(GL_TEXTURE_2D, skyboxTex[i]);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, 16.0f);
		
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
		glBindTexture(GL_TEXTURE_2D, 0);
	}
    //floorProg = new GLSLProgram(floorVS, floorPS);	
}



// TEXTURE STUFF BELOW
// (Taken from the NVIDIA CUDA SDK)

//------------------------------------------------------------------------------
// Function     	  : EvalHermite
// Description	    : 
//------------------------------------------------------------------------------
// EvalHermite(float pA, float pB, float vA, float vB, float u)
// Evaluates Hermite basis functions for the specified coefficients.
// 
inline float ParticleRenderer::evalHermite(float pA, float pB, float vA, float vB, float u)
{
    float u2=(u*u), u3=u2*u;
    float B0 = 2*u3 - 3*u2 + 1;
    float B1 = -2*u3 + 3*u2;
    float B2 = u3 - 2*u2 + u;
    float B3 = u3 - u;
    return( B0*pA + B1*pB + B2*vA + B3*vB );
}


unsigned char* ParticleRenderer::createGaussianMap(int N)
{
    float *M = new float[2*N*N];
    unsigned char *B = new unsigned char[4*N*N];
    float X,Y,Y2,Dist;
    float Incr = 2.0f/N;
    int i=0;  
    int j = 0;
    Y = -1.0f;
    //float mmax = 0;
    for (int y=0; y<N; y++, Y+=Incr)
    {
        Y2=Y*Y;
        X = -1.0f;
        for (int x=0; x<N; x++, X+=Incr, i+=2, j+=4)
        {
            Dist = (float)sqrtf(X*X+Y2);
            if (Dist>1) Dist=1;
            M[i+1] = M[i] = evalHermite(1.0f,0,0,0,Dist);
            B[j+3] = B[j+2] = B[j+1] = B[j] = (unsigned char)(M[i] * 255);
        }
    }
    delete [] M;
    return(B);
}    

void ParticleRenderer::_createTexture(int resolution)
{
    unsigned char* data = createGaussianMap(resolution);
    glGenTextures(1, (GLuint*)&m_texture);
    glBindTexture(GL_TEXTURE_2D, m_texture);
    glTexParameteri(GL_TEXTURE_2D, GL_GENERATE_MIPMAP_SGIS, GL_TRUE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, resolution, resolution, 0, 
                 GL_RGBA, GL_UNSIGNED_BYTE, data);
    
}

GLuint ParticleRenderer::createTexture(GLenum target, GLint internalformat, GLenum format, int w, int h, void *data)
{
    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(target, tex);
    glTexParameteri(target, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(target, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(target, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(target, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(target, GL_GENERATE_MIPMAP_SGIS, GL_TRUE);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(target, 0, internalformat, w, h, 0, format, GL_UNSIGNED_BYTE, data);
    return tex;
}

GLuint ParticleRenderer::loadTexture(char *filename)
{
    unsigned char *data = 0;
    unsigned int width, height;
    cutilCheckError( cutLoadPPM4ub(filename, &data, &width, &height));
    if (!data) {
        printf("Error opening file '%s'\n", filename);
        return 0;
    }
    printf("Loaded '%s', %d x %d pixels\n", filename, width, height);

    return createTexture(GL_TEXTURE_2D, GL_RGBA8, GL_RGBA, width, height, data);
}
