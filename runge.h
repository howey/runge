#define HEIGHT 20 //Atoms in y direction
#define WIDTH 20 //Atoms in x direction
#define DEPTH 20 //Atoms in z direction

#define BLOCK_SIZE 5 //Side length of 3D CUDA block
//#define VECTOR_SIZE 256 //Length of 1D CUDA block

#define BENCHMARK 1 //Time the simulation

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
double nextGaussian(void);
void computeField(Vector *, const SphVector *, int, Vector, Vector *);
void rk4(SphVector *, SphVector *, int, double, SphVector *, Vector *, Vector, Vector *); 
void mDot(SphVector *, SphVector *, int, Vector *);
void rkdumb(SphVector *, int, double, double, int, double *, SphVector **, Vector);
