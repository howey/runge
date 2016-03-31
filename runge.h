#define HEIGHT 2 //Atoms in y direction
#define WIDTH 2 //Atoms in x direction
#define DEPTH 1 //Atoms in z direction
#define SIZE (HEIGHT * WIDTH * DEPTH)

#define BLOCK_SIZE 5 //Side length of 3D CUDA block
//#define VECTOR_SIZE 256 //Length of 1D CUDA block

#define BENCHMARK 1 //Time the simulation

#define gaussian(mean, sd) ((mean) + (sd) * nextGaussian())

#define RIGID //Perform a rigid particle simulation

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
double nextGaussian(void);
void computeField(Vector *, const SphVector *, Vector, Vector *);
void rk4(SphVector *, SphVector *, double, SphVector *, Vector *, Vector, Vector *); 
void mDot(SphVector *, SphVector *, Vector *);
void rkdumb(SphVector *, double, double, int, double *, SphVector **, Vector);
