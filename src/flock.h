#include <cutil_inline.h>
#include <time.h>

#if defined(__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif

#include <cuda_gl_interop.h>

#include "textures.h"

class FlockSystem {
	public:
		FlockSystem(uint3 dim, uint N);
		~FlockSystem();		
		
		void resetBoids_Origin();
		void resetBoids_Centre();
		void resetBoids_Random();
		void toggleModelFlocking(bool);
		void toggleModelFlocking();
		void togglePictureFlocking();
		void togglePictureFlocking(bool);
		void updateSimulation();		
		void display();
		
		uint getNumParticles() const { return m_numParticles; }
	    unsigned int getCurrentReadBuffer() const { return m_vbo_pos; }
	    unsigned int getColorBuffer()       const { return m_vbo_col; }
	    float* getTarget();
		uint3 getGridDim() const { return m_gridSize; }
		float getParticleRadius() { return m_particleRadius; }		
		void setParticleSize(float radius){ m_particleRadius = radius; }
		void setGridSize(float x, float y, float z){ m_gridSize.x = (uint)x; m_gridSize.y = (uint)y; m_gridSize.z = (uint)z; }

		GLuint loadTexture(char *filename);
		GLuint createTexture(GLenum target, GLint internalformat, GLenum format, int w, int h, void *data);

		float* getModelVertices();
		//uint getNumParticles();

	protected:
		void initBuffers();
		uint createVBO(uint size, const GLvoid *data);
		void updateVBO(GLuint vbo, uint size, const GLvoid *data);
		void initCudaArrays();
		void initTargets();
		void setTarget();

	private:	
		// For picture flocking		
		bool pictureFlocking;
		GLuint picTex;	
		int picWidth;
		int picHeight;	
		void textureToBuffer();
		void paintFlock(float* color);
		
		// For model flocking
		bool modelFlocking;
	
		// Host Data
		float m_target[3];
		int m_numTargets;
		int m_currentTarget;
		float* m_targets;
		uint m_numParticles;
		uint3 m_gridSize;
		
		float m_simTime;
		float m_deltaTime;
		
		GLuint pbo[2];              // Position & Color
		uint   m_vbo_pos;           // vertex buffer object for particle positions
	    uint   m_vbo_pos_old;
	    uint   m_vbo_col;           // vertex buffer object for colors
	    uint   m_vbo_target;		// vertex buffer object for model vertices
		float *h_vbo_pos; 
		float *h_vbo_pos_old;
		float *h_vbo_col;
		float *h_vbo_target;	
		
		// Params
		float m_particleRadius;
		
		// GPU Data
		uint currentBuffer;
		float3 *d_vbo_pos; // to refer to position VBO
		float3 *d_vbo_pos_old;
		float4 *d_vbo_col; // to refer to colour VBO		
		float *vel; // [x1,x2,x3,...,y1,y2,y3,...,z1,z2,z3]
		float3 d_vbo_target;
		
		// OpenGL-CUDA interoperability
		struct cudaGraphicsResource *d_vbo_pos_resource; 
		struct cudaGraphicsResource *d_vbo_pos_old_resource; 
	    struct cudaGraphicsResource *d_vbo_col_resource; 	
	    struct cudaGraphicsResource *d_vbo_target_resource; 
	

};
