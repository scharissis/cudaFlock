// OpenGL/GLEW/GLUT
#include <GL/glew.h>
#if defined(__APPLE__) || defined(__MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif
#if defined (_WIN32)
#include <GL/wglew.h>
#endif


// Thrust
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>

#include <iostream>

// FlockSystem
#include "flock.h"
#include "flock.cuh"

// Flock model (TODO: hacky)
#include "load3ds/KDScene3DS.h"
KDScene3DS sc1;

// TODO: Duplicate in render_particles.cpp
GLuint FlockSystem::createTexture(GLenum target, GLint internalformat, GLenum format, int w, int h, void *data)
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

// TODO: Duplicate in render_particles.cpp
GLuint FlockSystem::loadTexture(char *filename)
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


// TODO: Handle arbitrary image sizes
void FlockSystem::textureToBuffer(){
	int w,h,size;	
	
//	picTex = loadTexture("../src/data/mona_64.ppm"); 
	picTex = loadTexture("../src/data/mona_64.ppm"); 
	
	glBindTexture(GL_TEXTURE_2D, picTex);
	
	glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH, &w);
	glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT, &h);
	picWidth = w;
	picHeight = h;
	std::cout << "Flock image is: " << w << " x " << h << std::endl;	
	
	float *data;
	size = w*h*4*sizeof(GLubyte); 	// sizeof(GLubyte) == 1		
	data = (float *)malloc(size*sizeof(float));	
	glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT, data);		
	
	// Reverse array (flip image upside down)
	float *data2 = (float *)malloc(size*sizeof(float));	
	for (int i=0; i<size/4; ++i){
		data2[i*4] = data[size-i*4];
		data2[i*4+1] = data[size-i*4+1];
		data2[i*4+2] = data[size-i*4+2];
		data2[i*4+3] = data[size-i*4+3];
	}
	
	
	// if (w*h > m_numParticles){	
	
	glBindBufferARB(GL_ARRAY_BUFFER_ARB, m_vbo_col);
	glBufferSubData(GL_ARRAY_BUFFER, 0, m_numParticles*4*sizeof(float), data2);
	glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);
	
	glBindTexture(GL_TEXTURE_2D, 0);
	
	free(data);
}

void FlockSystem::initBuffers(){
	/*******************/	
	// Model To Buffer (in progress)
	// WARNING: If you use less particles than vertices it will cause a memory corruption (and segfault)
	/*******************/	
	
	// Flock Model
	//sc1.load3DS("../src/load3ds/models/hornet_jet/hornet.3ds");	 // Segfault
	//sc1.load3DS("../src/load3ds/models/p.3ds");	 // 8046 verts	 // parts are dislocated
	//sc1.load3DS("../src/load3ds/models/plane/a36.3ds");		// 13544 verts
	sc1.load3DS("../src/load3ds/models/goldfish.3ds");		// verts: 17945		// GOOD <---
	//sc1.load3DS("../src/load3ds/models/goblin.3ds");		// verts: 12829	
	//sc1.load3DS("../src/load3ds/models/gallardo/gred.3ds");		// verts: 135154 // just see part of it
	//sc1.load3DS("../src/load3ds/models/irobot.3ds");		// verts: 103497  // just see part of it
	//sc1.load3DS("../src/load3ds/models/rudfish.3ds");		// verts: 1450  // looks alright; needs to be rotated :(			
	//sc1.load3DS("../src/load3ds/models/minifish.3ds");		// verts: 202   // looks ok; needs to be rotated :(
	//sc1.load3DS("../src/load3ds/models/dolphin_low.3ds");	// verts: 212   // too small?
	//sc1.load3DS("../src/load3ds/models/fish.3ds");	// verts: 4386	 // GOOD
	//sc1.load3DS("../src/load3ds/models/piranha.3ds");	//	verts: 18430  // points downwards
		
	int totalVerts = 0;	
	for (vector<KDObject>::iterator iter = sc1.objects.begin(); iter != sc1.objects.end(); iter++) {
			totalVerts += iter->number_of_vertices;
	}
	std::cout << "\t== verts: " << totalVerts << std::endl;
	
	if (m_numParticles < totalVerts){
		m_numParticles = totalVerts;
		/*
		int powTwo[10] = {512,1024,2048,4096,8192,16384,32768,65536,131072,262144}; // 10 of them
		for (int n=0;n<10;++n){
			if (powTwo[n] > totalVerts) {
				m_numParticles = powTwo[n];
				return;
			}			
		}
		if (m_numParticles < totalVerts){ // if loop above did nothing
				std::cerr << "Error: number of particles must be a power of 2." << std::endl;
		}
		*/
		std::cout << "WARNING: number of particles increased to " << m_numParticles << "." << std::endl;
	}


	float m_baseColor[4] = { 0.6f, 0.1f, 0.0f, 0.95f};
	uint size = 0;
		
    thrust::host_vector<float> h_vec(m_numParticles*3);
    thrust::generate(h_vec.begin(), h_vec.end(), rand);    
	std::vector<float> h_vec_stl(h_vec.size());
	thrust::copy(h_vec.begin(), h_vec.end(), h_vec_stl.begin());

	h_vbo_pos = new float[m_numParticles*3];
    memset(h_vbo_pos, 0, m_numParticles*3*sizeof(float));    
    h_vbo_pos_old = new float[m_numParticles*3];
    memset(h_vbo_pos_old, 0, m_numParticles*3*sizeof(float)); 
    h_vbo_col = new float[m_numParticles*4];
    memset(h_vbo_col, 0, m_numParticles*4*sizeof(float));      
	h_vbo_target = new float[m_numParticles*3];
    memset(h_vbo_target, 0, m_numParticles*3*sizeof(float));    
	
	// copy a device_vector into an STL vector
	//thrust::device_vector<int> D(stl_list.begin(), stl_list.end());
    //std::vector<int> stl_vector(D.size());
    //thrust::copy(D.begin(), D.end(), stl_vector.begin());

	
	for (uint i=0; i<m_numParticles; ++i){		
		h_vbo_pos_old[i*3+0] = h_vbo_pos[i*3+0] = (int)h_vec_stl[i*3+0]%(m_gridSize.x);
		h_vbo_pos_old[i*3+0] = h_vbo_pos[i*3+1] = (int)h_vec_stl[i*3+1]%(m_gridSize.y);
		h_vbo_pos_old[i*3+0] = h_vbo_pos[i*3+2] = (int)h_vec_stl[i*3+2]%(m_gridSize.z);		
	}
	
	for (uint i=0; i<m_numParticles; ++i){
     	/*
     	h_vbo_col[i*4+0] = (float)((int)h_vec_stl[i*3+0]%(m_gridSize.x))/m_gridSize.x;
    	h_vbo_col[i*4+1] = (float)((int)h_vec_stl[i*3+1]%(m_gridSize.y))/m_gridSize.y;
    	h_vbo_col[i*4+2] = (float)((int)h_vec_stl[i*3+2]%(m_gridSize.x))/m_gridSize.z;
    	*/
    	h_vbo_col[i*4+0] = m_baseColor[0];
    	h_vbo_col[i*4+1] = m_baseColor[1];
    	h_vbo_col[i*4+2] = m_baseColor[2];
    	h_vbo_col[i*4+3] = m_baseColor[3];
    }
		
	float scale = 2.0f;	
	float xOff = m_gridSize.x/2.0f;	
	float yOff = m_gridSize.y/2.0f;
	float zOff = m_gridSize.z/2.0f;
		
	int v = 0;	
	for (vector<KDObject>::iterator iter = sc1.objects.begin(); iter != sc1.objects.end(); iter++) {
		
		for (int i=0; i<iter->number_of_vertices; i++)
		{
			float x1, y1, z1;
			x1 = iter->vertices[i].position.x;// - iter->object_matrix._41; 
			y1 = iter->vertices[i].position.y;// - iter->object_matrix._42; 
			z1 = iter->vertices[i].position.z;// - iter->object_matrix._43; 		
			//std::cout << "(" << x1 << "," << y1 << "," << z1 << ")" << std::endl;
			h_vbo_target[v*3+0] = scale * x1 + xOff;
			h_vbo_target[v*3+1] = scale * y1 + yOff;
			h_vbo_target[v*3+2] = scale * z1 + zOff;
			v++;
		}		
	}	
	
	/*******************/	
	
	// Initialise Buffers and Register them with CUDA
	
	float memNeeded = (sizeof(float)*m_numParticles*(3+3+3+4))/1024;
	std::cout << "Device buffer memory usage: " << memNeeded << " MB" << std::endl;
	
	// Position Buffers
	size = 3 * sizeof(float) * m_numParticles;
	m_vbo_pos = createVBO(size, h_vbo_pos); 
	registerGLBufferObject(m_vbo_pos,&d_vbo_pos_resource);

	size = 3 * sizeof(float) * m_numParticles;
	m_vbo_pos_old = createVBO(size, h_vbo_pos_old); 
	registerGLBufferObject(m_vbo_pos_old,&d_vbo_pos_old_resource);
	
	size = 3 * sizeof(float) * m_numParticles;
	m_vbo_target = createVBO(size, h_vbo_target); 
	registerGLBufferObject(m_vbo_target,&d_vbo_target_resource);
	
	// Color Buffer
	size = 4 * sizeof(float) * m_numParticles;
	m_vbo_col = createVBO(size, h_vbo_col);
	registerGLBufferObject(m_vbo_col,&d_vbo_col_resource);	
	
}

void FlockSystem::initCudaArrays(){
	// Velocity
	// Velocity array is 1D (x,y,z) for thrust:zip-ping for efficiency
	uint memSize = sizeof(float) * 3 * m_numParticles;
	allocateArray((void**)&vel, memSize);

}

void FlockSystem::initTargets(){
	float abit = m_gridSize.x/100.0f; // abit==sizeof(teapot)  [so it doesnt render outside of skybox...]

	m_numTargets = 4;
	m_target[0] = (float)m_gridSize.x/2.0f;
	m_target[1] = (float)m_gridSize.y/2.0f;
	m_target[2] = (float)m_gridSize.z/2.0f;
	m_currentTarget = 0;
	m_targets = (float *)malloc(m_numTargets * 3*sizeof(float));
	m_targets[0*3+0] = m_gridSize.x/2.0f; 	m_targets[0*3+1] = m_gridSize.y/2.0f; m_targets[0*3+2] = m_gridSize.z/2.0f;
	m_targets[1*3+0] = 0.0f + abit;			m_targets[1*3+1] =  m_gridSize.y - abit; 	  m_targets[1*3+2] = 0.0f + abit;
	m_targets[2*3+0] = m_gridSize.x - abit;		m_targets[2*3+1] = m_gridSize.y - abit; 	  m_targets[2*3+2] = m_gridSize.z - abit;
	m_targets[3*3+0] = 0.0f + abit;				m_targets[3*3+1] = 0.0f + abit; 			  m_targets[3*3+2] = 0.0f + abit;
}

FlockSystem::FlockSystem(uint3 dim, uint N){
	//std::cerr << "Creating new FlockSystem..." << std::endl;	// DEBUG
	std::cerr << "NUM P == " << N << std::endl;
	m_numParticles = N;
	m_gridSize.x = dim.x;
	m_gridSize.y = dim.y; 
	m_gridSize.z = dim.z;
	m_simTime = 0.0f;
	m_deltaTime = 0.01f;
	
	m_particleRadius = m_gridSize.x*0.001f;
	
	currentBuffer = 1; // dPos is the one that has values initialised...
	
	pictureFlocking = false;
	modelFlocking = false;
	
	initBuffers();
	initCudaArrays();
	initTargets();
}

FlockSystem::~FlockSystem(){
	//std::cerr << "Deleting FlockSystem..." << std::endl;
	
	 if (picTex)  glDeleteTextures(1, &picTex);
	
	delete [] h_vbo_pos;	
	delete [] h_vbo_pos_old;
	delete [] h_vbo_col;
	delete [] h_vbo_target;
	
	unregisterGLBufferObject(d_vbo_pos_resource);
	unregisterGLBufferObject(d_vbo_col_resource);
	glDeleteBuffers(1, (const GLuint*)&pbo[0]);
	glDeleteBuffers(1, (const GLuint*)&pbo[1]);
}

uint FlockSystem::createVBO(uint size, const GLvoid *data)
{
    GLuint vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, size, data, GL_DYNAMIC_DRAW);
    
    // Check data was allocated (DEBUG)
	int s = 0;
		glGetBufferParameteriv(GL_ARRAY_BUFFER, GL_BUFFER_SIZE, (GLint*)&s);     
		if ((unsigned)s != size)
			fprintf(stderr, "WARNING: Buffer Object allocation FAILED!\n");

    
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    return vbo;
}

void FlockSystem::resetBoids_Origin(){
	float *dPos = (float *) mapGLBufferObject(&d_vbo_pos_resource);
	set_boids_toOrigin(dPos, m_numParticles);
	unmapGLBufferObject(d_vbo_pos_resource);
}

void FlockSystem::resetBoids_Centre(){
	float *dPos = (float *) mapGLBufferObject(&d_vbo_pos_resource); 	
	const float x = (float)m_gridSize.x/2.0f;
	const float y = (float)m_gridSize.y/2.0f;
	const float z = (float)m_gridSize.z/2.0f;
	set_boids_toPoint(dPos, m_numParticles, x,y,z);
	unmapGLBufferObject(d_vbo_pos_resource);
}

// TODO: Doesn't work
void FlockSystem::resetBoids_Random(){
	//float *dPos = (float *) mapGLBufferObject(&d_vbo_pos_resource); 
	//thrust::device_ptr<float> tdPos((float *)dPos);	
	//thrust::generate(tdPos,tdPos+m_numParticles*3,rand);	
	//unmapGLBufferObject(d_vbo_pos_resource);
	
	thrust::host_vector<float> h_vec(m_numParticles*3);
    thrust::generate(h_vec.begin(), h_vec.end(), rand);    
	std::vector<float> h_vec_stl(h_vec.size());
	thrust::copy(h_vec.begin(), h_vec.end(), h_vec_stl.begin());
	
	float* data = (float*)malloc(m_numParticles*3*sizeof(float));
	
	for (uint i=0; i<m_numParticles; ++i){		
		data[i*3+0] = (int)h_vec_stl[i*3+0]%(m_gridSize.x);
		data[i*3+1] = (int)h_vec_stl[i*3+1]%(m_gridSize.y);
		data[i*3+2] = (int)h_vec_stl[i*3+2]%(m_gridSize.z);		
	}	
	
	glBindBufferARB(GL_ARRAY_BUFFER_ARB, m_vbo_pos);
	glBufferSubData(GL_ARRAY_BUFFER, 0, m_numParticles*3*sizeof(float), data);
	glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);	
	
	free(data);
}

void FlockSystem::setTarget(){  
	if ((int)m_simTime%3 == 2){
		m_currentTarget = (m_currentTarget+1)%m_numTargets;
	
		m_target[0] = m_targets[m_currentTarget*3+0];
		m_target[1] = m_targets[m_currentTarget*3+1];
		m_target[2] = m_targets[m_currentTarget*3+2];
		
		m_simTime = 0.0f; // TODO: Hacky
		//std::cerr << "Target: <<" << m_target[0] << "," << m_target[1] << "," << m_target[2] << ">>" << std::endl;	// DEBUG
	} 
}

float* FlockSystem::getTarget(){
	return m_target;
}

void FlockSystem::updateSimulation(){
	float *dPos = (float *) mapGLBufferObject(&d_vbo_pos_resource); 
	float *dPos_old = (float *) mapGLBufferObject(&d_vbo_pos_old_resource); 
	float *dTarget = (float *) mapGLBufferObject(&d_vbo_target_resource); 
	//float *dCol = (float *) mapGLBufferObject(&d_vbo_col_resource); 	
	
	setTarget();
	
	if (currentBuffer==0){
		move_boids(dPos_old, dPos, vel, dTarget, m_target, m_gridSize, m_particleRadius, m_numParticles, m_simTime, modelFlocking, pictureFlocking, picWidth, picHeight);
	} else {
		move_boids(dPos, dPos_old, vel, dTarget, m_target, m_gridSize, m_particleRadius, m_numParticles, m_simTime, modelFlocking, pictureFlocking, picWidth, picHeight);
	}
	
	m_simTime += m_deltaTime;
	currentBuffer = (currentBuffer+1)%2;
	
	unmapGLBufferObject(d_vbo_pos_resource);
	unmapGLBufferObject(d_vbo_pos_old_resource);
	unmapGLBufferObject(d_vbo_target_resource);
}

// Paint all boids 'color' (RGBA) by modifying underlying color buffer object
void FlockSystem::paintFlock(float* color){
	int size = m_numParticles*4;
	float* data = (float*)malloc(size*sizeof(float));
	
	for (int i=0; i<m_numParticles; ++i){
		data[i*4+0] = color[0];
		data[i*4+1] = color[1];
		data[i*4+2] = color[2];
		data[i*4+3] = color[3];
	}
	
	glBindBufferARB(GL_ARRAY_BUFFER_ARB, m_vbo_col);
	glBufferSubData(GL_ARRAY_BUFFER, 0, size*sizeof(float), data);
	glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);	
	
	free(data);
}

void FlockSystem::toggleModelFlocking(bool boolean){
	modelFlocking = boolean;
}

void FlockSystem::toggleModelFlocking(){
	modelFlocking = !modelFlocking;
	
	if(pictureFlocking) togglePictureFlocking();
}

void FlockSystem::togglePictureFlocking(bool boolean){
	pictureFlocking = boolean;
}

void FlockSystem::togglePictureFlocking(){
	if(pictureFlocking){
		float color[4] = { 0.6f, 0.1f, 0.0f, 0.95f};
		paintFlock(color);
		pictureFlocking = false;
	} else {
		textureToBuffer();
		pictureFlocking = true;
	}
	
	if (modelFlocking) toggleModelFlocking();
}

float* FlockSystem::getModelVertices(){
	return h_vbo_target;
}
