#define HEIGHT 23 //Atoms in y direction
#define WIDTH 23 //Atoms in x direction
#define DEPTH 23 //Atoms in z direction

#define BLOCK_SIZE 10 //Side length of 3D CUDA block
#define VECTOR_SIZE 256 //Length of 1D CUDA block

#define BENCHMARK 0 //Time the simulation

#define DEBUG 1 //Watch CUDA API calls for runtime errors

#define gaussian(mean, sd) ((mean) + (sd) * nextGaussian())

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>

//Type declarations

//A rectangular vector structure
typedef struct {
	double x;
	double y;
	double z;
} Vector;

//A spherical vector structure
typedef struct {
	double r;
	double theta;
	double phi;
} SphVector;

//Function prototypes

double nextDouble(double, double);
double nextGaussian();
//void mDot(double, SphVector[], SphVector[], int, Vector[]);
