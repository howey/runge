#define USE_THERMAL 0 //Simulate random thermal motion. 1 enables, 0 disables 

#define HEIGHT 2 //Atoms in y direction
#define WIDTH 2 //Atoms in x direction
#define DEPTH 2 //Atoms in z direction

#define BLOCK_SIZE 4 //Side length of CUDA block

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
void mDot(double, SphVector[], SphVector[], int, Vector[]);
