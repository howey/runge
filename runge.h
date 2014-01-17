#define USE_THERMAL 0 

#define HEIGHT 16 //Atoms in y direction
#define WIDTH 16 //Atoms in x direction
#define DEPTH 16 //Atoms in z direction

#define BLOCK_SIZE 4 //Side length of CUDA block

__global__ void mDot(double, SphVector, SphVector, int, Vector);
