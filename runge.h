#define USE_THERMAL 0 

#define HEIGHT 4 //Atoms in y direction
#define WIDTH 4 //Atoms in x direction
#define DEPTH 4 //Atoms in z direction

#define BLOCK_SIZE 4 //Side length of CUDA block

__global__ void mDot(double, SphVector, SphVector, int, Vector);
