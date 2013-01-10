// This file contains C wrappers around the some of the CUDA API and the
// kernel functions so that they can be called from "flock.cpp"


extern "C"
{

void registerGLBufferObject(uint vbo, struct cudaGraphicsResource **cuda_vbo_resource);
void unregisterGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource);
void *mapGLBufferObject(struct cudaGraphicsResource **cuda_vbo_resource);
void unmapGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource);
void allocateArray(void **devPtr, size_t size);
void freeArray(void *devPtr);
void initCudaGL(int argc, char **argv);

void set_boids_toOrigin(float *dPos, uint numParticles);
void set_boids_toCentre(float *dPos, uint dim, uint numParticles);
void set_boids_toPoint(float *dPos, const uint numParticles, const float x, const float y, const float z);
void move_boids(float *dPos_old, float *dPos, float *dVel, float *dTarget, float *target, uint3 gridSize, float pRadius, uint numParticles, float time, bool modelFlocking, bool pictureFlocking, int picWidth, int picHeight);
}
