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

#ifndef __RENDER_PARTICLES__
#define __RENDER_PARTICLES__

#include "textures.h"

class ParticleRenderer
{
public:
    ParticleRenderer(uint3 dim);
    ~ParticleRenderer();

    void setPositions(float *pos, int numParticles);
    void setVertexBuffer(unsigned int vbo, int numParticles);
    void setColorBuffer(unsigned int vbo) { m_colorVBO = vbo; }
    

    enum DisplayMode
    {
        PARTICLE_POINTS,
        PARTICLE_FIRE,
        SPRITE_COSMOS,
        PARTICLE_SPHERES,
        PARTICLE_NUM_MODES
    };

    void display();
    void setNextDisplayMode();
    void setDisplayMode(DisplayMode mode);
    void toggleSkyBox(bool);
    void toggleSkyBox();
    void toggleFloor(bool);
    void toggleFloor();

    void setParticleRadius(float r) { m_particleRadius = r; }
    void setFOV(float fov) { m_fov = fov; }
    void setWindowSize(int w, int h) { m_window_w = w; m_window_h = h; }

protected: // methods
    void _initGL();
    void _drawPoints(bool useColorVBO);
    void drawSky();
    void drawFloor();
    void drawSkyBox();
    GLuint _compileProgram(const char *vsource, const char *fsource);
    
    inline float evalHermite(float pA, float pB, float vA, float vB, float u);
    unsigned char* createGaussianMap(int N);
    void _createTexture(int resolution);
    GLuint createTexture(GLenum target, GLint internalformat, GLenum format, int w, int h, void *data);
    GLuint loadTexture(char *filename);

protected: // data
    float *m_pos;
    uint3 m_gridDim;
    int m_numParticles;

    float m_particleRadius;
    float m_fov;
    int m_window_w, m_window_h;

    GLuint m_program_sphere;
    GLuint m_program_cosmos;		
	unsigned int m_texture; // sprite fire texture
	float m_spriteSize;     // also for fire

    GLuint m_vbo;
    GLuint m_colorVBO;
    
private:	
	GLuint floorTex;
	GLuint skyboxTex[6];
	bool b_renderSkyBox;
	bool b_renderFloor;
	DisplayMode _displayMode;
};

#endif //__ RENDER_PARTICLES__
