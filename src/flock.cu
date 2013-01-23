#include <cutil_inline.h>
#include <cutil_math.h>

#include <cuda_gl_interop.h>

// Thrust
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>

// Macros to simplify shared memory addressing
//#define SX(i) sharedPos[i+blockDim.x*threadIdx.y]  // nbody2
#define SX(i) smemPos[i+blockDim.x]


extern "C"
{

void registerGLBufferObject(uint vbo, struct cudaGraphicsResource **cuda_vbo_resource) {
	//std::cerr << "RegisteringGLBuffer()..." << std::endl;
	//std::cerr << "\t" << cudaGetErrorString(cudaGetLastError()) << std::endl;	// DEBUG
    cutilSafeCall(cudaGraphicsGLRegisterBuffer(cuda_vbo_resource, vbo, cudaGraphicsMapFlagsNone));
}

void unregisterGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource) {
    cutilSafeCall(cudaGraphicsUnregisterResource(cuda_vbo_resource));	
}

void *mapGLBufferObject(struct cudaGraphicsResource **cuda_vbo_resource) {
	//std::cerr << "\t" << cudaGetErrorString(cudaGetLastError()) << std::endl;	// DEBUG
	void *ptr;
	cutilSafeCall(cudaGraphicsMapResources(1, cuda_vbo_resource, 0));
	size_t num_bytes; 
	cutilSafeCall(cudaGraphicsResourceGetMappedPointer((void **)&ptr, &num_bytes, *cuda_vbo_resource));
	return ptr;
}

void unmapGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource) {
	cutilSafeCall(cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0));
}

void allocateArray(void **devPtr, size_t size) {
	cutilSafeCall(cudaMalloc(devPtr, size));
	cutilSafeCall(cudaMemset(*devPtr,0,size));
}

void freeArray(void *devPtr) {
    cutilSafeCall(cudaFree(devPtr));
}

void initCudaGL(int argc, char **argv, int numDevs) {
	for (int i=0; i<numDevs; ++i){
		cutilSafeCall(cudaSetDevice(i));
        cutilSafeCall(cudaSetDeviceFlags(cudaDeviceMapHost));	
	}
	
	// Set device for OpenGL Interoperability and Rendering
	if ( cutCheckCmdLineFlag(argc, (const char **)argv, "device")) {
		cutilDeviceInit (argc, argv);
	} else {
		cudaGLSetGLDevice (cutGetMaxGflopsDeviceId() );
	}	
}

float3 centreOfMass(float *pos, uint numParticles) {
    thrust::device_ptr<float> dPos((float *)pos);

	uint N = numParticles;
	float Nf = (float)N;

    float x = thrust::reduce(dPos,dPos+N)/Nf;
    float y = thrust::reduce(dPos+N,dPos+2*N)/Nf;
    float z = thrust::reduce(dPos+2*N,dPos+3*N)/Nf;
    
    return make_float3(x,y,z);
}

float3 meanVelocity(float *vel, uint numParticles) {
    thrust::device_ptr<float> dVel((float *)vel);

	uint N = numParticles;
	float Nf = (float)N;
	
    float x = thrust::reduce(dVel,dVel+N)/Nf;
    float y = thrust::reduce(dVel+N,dVel+2*N)/Nf;
    float z = thrust::reduce(dVel+2*N,dVel+3*N)/Nf;
    
    return make_float3(x,y,z);
}


// 3D Euclidian Distance
inline __host__ __device__ float distance3(float3 a, float3 b) {
	return sqrt(powf((a.x-b.x),2.0f)+powf((a.y-b.y),2.0f)+powf((a.z-b.z),2.0f));
}

// Magnitude
inline __host__ __device__ float magnitude(float3 a) {
	return sqrt(powf(distance3(make_float3(0.0f,0.0f,0.0f),a),2));
}

// Normalise (Get a unit vector)
inline __host__ __device__ float3 normalise(float3 a) {
	return (a/magnitude(a));
}

// Rule 1: Boids try to fly towards the centre of mass of neighbouring boids
__device__ float3 rule1(float3 pos, float3 meanPos, uint numParticles) {
	float3 pc = (meanPos*numParticles - pos)/(float)(numParticles-1);
	return (pc-pos)/100.0f;
}

// Rule 2: Boids try to keep a small distance away from other objects (including other boids)
__device__ float3 rule2_old(float3 pos, float3 *dPos, float distance, uint numParticles) {
	float3 c = make_float3(0.0f,0.0f,0.0f);	
		
	for (uint b=0; b<numParticles; ++b){	
			float3 bp = dPos[b];			
			if (distance3(pos,bp) < distance){
				c = c - (bp-pos);
			}		
	}	
	return c;
}

__device__ float3 tile_calculation(float3 pos, float d) {
	extern __shared__ float3 smemPos[];
	unsigned long j = 0;
	float3 c = make_float3(0.0f,0.0f,0.0f);	
	
	#pragma unroll 8
	for (int i=0; i<blockDim.x; ++i){
		float3 spos = smemPos[j++];
		//float3 spos = SX(j++);
		if (distance3(pos,spos) < d){
				c = c - (spos-pos);
		}
	}
	return c;
}

// WRAP is used to force each block to start working on a different 
// chunk (and wrap around back to the beginning of the array) so that
// not all multiprocessors try to read the same memory locations at 
// once.
#define WRAP(x,m) (((x)<m)?(x):(x-m))  // Mod without divide, works on values from 0 up to 2m

// Rule 2: Boids try to keep a small distance away from other objects (including other boids)
// TILE_SIZE should be equal to WARP_SIZE?
#define TILE_SIZE 256
__device__ float3 rule2(float3 pos, float3 *dPos, float distance, uint numParticles){
	float3 c = make_float3(0.0f,0.0f,0.0f);	
	extern __shared__ float3 smemPos[];
	float3 tmpP;
	float x,y,z;
	float3 bp;
	uint i = 0;
	for (uint b=0; b<numParticles; ){			     
		i = threadIdx.x;
		smemPos[i] = dPos[b+i];

		__syncthreads();
		
		#pragma unroll 8
		for (i=b; i < b+TILE_SIZE ; i++) {
			bp = smemPos[i-b];			
			tmpP = pos;
				
			x = (tmpP.x-bp.x); 
			y = (tmpP.y-bp.y); 
			z = (tmpP.z-bp.z); 

			if (x*x+y*y+z*z < distance*distance){
				c = c - (bp-pos);
			}		
		}
		b = b+TILE_SIZE;
		
		__syncthreads();
	}	
	return c;
}

// TODO: Buggy! :/
__device__ float3 rule2_v2(float3 pos, float3 *dPos, float distance, uint numParticles) {
	float3 p = make_float3(0.0f,0.0f,0.0f);
		
	extern __shared__ float3 smemPos[];
	int i, tile;
	
	for (i = 0, tile = 0; i < numParticles; i += blockDim.x, tile++) {
		smemPos[threadIdx.x] = dPos[WRAP(blockIdx.x + tile, gridDim.x) * blockDim.x + threadIdx.x];
		__syncthreads();	
		
		p = tile_calculation(pos, distance);		
	
		__syncthreads();
	}
	
	return p;
}

// Rule 3: Boids try to match velocity with near boids
__device__ float3 rule3(float3 vel, float3 meanVel, uint numParticles) {
	float3 pv = (meanVel*numParticles - vel)/(float)(numParticles-1);
	return (pv-vel)/8.0f;
}

// Rule 4: Tendency towards a particular target place
__device__ float3 rule4(float3 target, float3 pos) {
	return (target-pos)/100.0f;
}

// Rule 5: Tendency away from a particular target place
__device__ float3 rule5(float3 target, float3 pos) {
	return -1.0f*rule4(target,pos);
}

// Limit speed
__device__ float3 limit_speed(float3 v, float maxSpeed){
	float m = magnitude(v);
	if (m > maxSpeed){
		return ((v/m)*maxSpeed);
	}	
	return v;
}

// Bound Position
__device__ float3 bound_position(float3 p, float minX, float maxX, float minY, float maxY, float minZ, float maxZ) {
	float3 v = p;
	const float f = 10.0f;
	
	if (p.x < minX){
		v.x = f;
	} else if (p.x > maxX){
		v.x = -f;
	}
	
	if (p.y < minY){
		v.y = f;
	} else if (p.y > maxY){
		v.y = -f;
	}
	
	if (p.z < minZ){
		v.z = f;
	} else if (p.z > maxZ){
		v.z = -f;
	}
	return v;
}

__global__ void step(float3 *dPos_old, float3 *dPos, float3 *dVel, float3 *dTarget, float3 pc, float3 pv, float3 target, float3 gridSize, float pRadius, uint numParticles, float time, bool modelFlocking, bool pictureFlocking, int picWidth, int picHeight) {
	uint gid = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	float3 v1 = make_float3(0.0f,0.0f,0.0f);
	float3 v2 = make_float3(0.0f,0.0f,0.0f);
	float3 v3 = make_float3(0.0f,0.0f,0.0f);
	float3 v4 = make_float3(0.0f,0.0f,0.0f);
	float3 v5 = make_float3(0.0f,0.0f,0.0f);

	if (gid >= numParticles) return;
	
	// TODO: Bug in rule_2 (from 17945 onwards which is the size of the model - so perhaps it's related to that?)
	numParticles -= 512; //*=(3.f/4.f); 
	
	float3 vel = dVel[gid];
	float3 pos = dPos_old[gid];		

	float maxSpeed = gridSize.z/250.0f;
	float slowMo = maxSpeed * (1.f/2.f);

	if (pictureFlocking){
		int width = picWidth;
		int height = picHeight;
		int separation = pRadius*2;
		float xOff = gridSize.x/2-separation*width/2;
		float zOff = gridSize.z/2-separation*height/2;
		if (gid <= width*height){
			target = make_float3(xOff+gid%width*separation, gridSize.y/2, zOff+(gid/height+1)*separation);
		} else {
			target = make_float3(0,0,0);
		}		
		maxSpeed = slowMo;
	}

	if (modelFlocking){
		target = dTarget[gid];
		maxSpeed = slowMo;
	}

	float distance = pRadius*2.0f*4.0f; // TODO: Put into __constant__ memory	
	float m1 = 1.0f; // Scatters the flock if negative	
	
	if (!pictureFlocking && !modelFlocking){
		v1 = m1 * rule1(pos, pc, numParticles);	
		v2 = rule2(pos, dPos, distance, numParticles);     
	}
	v3 = rule3(vel, pv, numParticles); 	
	v4 = rule4(target, pos);
	//v5 =      rule5(v5,p);

	vel += (v1 + v2 + v3 + v4);
	vel = limit_speed(vel,maxSpeed);
	pos = bound_position(pos,0.0f,gridSize.x,0.0f,gridSize.y,0.0f,gridSize.z);	
	
	dVel[gid] = vel;
	dPos[gid] = pos + vel;	
}

__global__ void setPoints(float3 *dPos, float3 p, uint N) {
	uint gid = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (gid < N) dPos[gid] = p;
}

//Round a / b to nearest higher integer value
inline uint iDivUp(uint a, uint b) {
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

// compute grid and thread block size for a given number of elements
void computeGridSize(uint n, uint blockSize, uint &numBlocks, uint &numThreads) {
    numThreads = min(blockSize, n);
    numBlocks = iDivUp(n, numThreads);
}

void set_boids_toOrigin(float *dPos, uint numParticles) {
	cudaMemset(dPos,0,numParticles*3*sizeof(float));
}

void set_boids_toPoint(float *dPos, const uint numParticles, const float x, const float y, const float z) {
	uint numThreads, numBlocks;
	const uint maxThreads = 256;
    computeGridSize(numParticles, maxThreads, numBlocks, numThreads);
    setPoints<<<numBlocks,numThreads>>>((float3 *)dPos, make_float3(x,y,z), numParticles);
}

void move_boids(float *dPos_old, float *dPos, float *dVel, float *dTarget, float *t, uint3 ugrid, float pRadius, uint numParticles, float time, bool modelFlocking, bool pictureFlocking, int picWidth, int picHeight) {	
	float3 pc = centreOfMass(dPos, numParticles); // Perceived Centre of mass
	float3 pv = meanVelocity(dVel, numParticles); // Perceived Velocity
	float3 grid = make_float3((float)ugrid.x,(float)ugrid.y,(float)ugrid.z);
	float3 target = make_float3(t[0],t[1],t[2]); // TODO: Use dTarget
	
	uint nThreads = 256;
	uint numBlocks = (numParticles%nThreads==0)?numParticles/nThreads:numParticles/nThreads+1;

	int smem = TILE_SIZE * 3 * sizeof(float3); // Bytes of Shared Memory Required
	step<<<numBlocks,nThreads,smem>>>((float3 *)dPos_old, (float3 *)dPos, (float3 *)dVel, (float3 *)dTarget, pc, pv, target, grid, pRadius, numParticles, time, modelFlocking, pictureFlocking, picWidth, picHeight);
}

} // extern "C"
