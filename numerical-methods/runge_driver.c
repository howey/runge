#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include "vector.h"
#include "runge.h"
#include "mars.h"

static const double ALPHA = 35186; //alpha*gamma / (1 + alpha^2)
static const double GAMMA = 1.76e7;
static const double K = 1e6;
static const double TIMESTEP = 2.5e-12;

static double *xx;
static SphVector **y;
static Vector H;

#if 0 
//Evaluates dthetadt[0..n-1] at t and theta[0..n-1]
//This particular derivative is not a function of t
void derivative(double t, double theta[], double dthetadt[], int n) {
	for(int i = 0; i < n; i++) {
		dthetadt[i] = (-2.2067e-12)*(pow(theta[i], 4) - 81e8);
	}
}
#endif

//Computes the anisotropy field and writes the result to a Vector H
void anisotropyH(Vector * H, const SphVector * M) {
	H->x = (1/M->r) * -2 * K * cos(M->theta) * sin(M->theta) * cos(M->phi) * cos(M->theta);
	H->y = (1/M->r) * -2 * K * cos(M->theta) * sin(M->theta) * sin(M->phi) * cos(M->theta);
	H->z = (1/M->r) * 2 * K * cos(M->theta) * sin(M->theta) * sin(M->theta);
}

void mDot(double t, SphVector M[], SphVector dMdt[], int n) {
	for(int i = 0; i < n; i++) {
		dMdt[i].phi = GAMMA * ((cos(M[i].theta) * sin(M[i].phi) * H.y) / sin(M[i].theta) + (cos(M[i].theta) * cos(M[i].phi) * H.x) / sin(M[i].theta) - H.z) + ALPHA * ((cos(M[i].phi) * H.y) / sin(M[i].theta) - (sin(M[i].phi) * H.x) / sin(M[i].theta));
		dMdt[i].theta = -GAMMA * (cos(M[i].phi) * H.y - sin(M[i].phi) * H.x) + ALPHA * (cos(M[i].theta) * cos(M[i].phi) * H.x - H.z * sin(M[i].theta) + cos(M[i].theta) * sin(M[i].phi) * H.y);
	}
}

#if 0
//Evaluates dphidt[0..n-1] at t, theta[0..n-1] and phi[0..n-1]
void phiDot(double t, double theta[], double phi[], double dphidt[], const Vector * H, int n) {
	for(int i = 0; i < n; i++) {
		dphidt[i] = GAMMA * ((cos(theta[i]) * sin(phi[i]) * H->y) / sin(theta[i]) + (cos(theta[i]) * cos(phi[i]) * H->x) / sin(theta[i]) - H->z) + ALPHA * ((cos(phi[i]) * H->y) / sin(theta[i]) - (sin(phi[i]) * H->x) / sin(theta[i]));	
	}
}

//Evaluates dthetadt[0..n-1] at t, theta[0..n-1] and phi[0..n-1]
void thetaDot(double t, double theta[], double phi[], double dthetadt[], const Vector * H, int n) {
	for(int i = 0; i < n; i++) {
		dthetadt[i] = -GAMMA * (cos(phi[i]) * H->y - sin(phi[i]) * H->x) + ALPHA * (cos(theta[i]) * cos(phi[i]) * H->x - H->z * sin(theta[i]) + cos(theta[i]) * sin(phi[i]) * H->y);
	}
}
#endif

/*
Starting from initial values vstart[0..nvar-1] known at x1 use fourth-order Runge-Kutta
to advance nstep equal increments to x2. The user-supplied routine derivs(x,v,dvdx)
evaluates derivatives. Results are stored in the global variables y[0..nvar-1][0..nstep]
and xx[0..nstep].
*/
void rkdumb(SphVector vstart[], int nvar, double x1, double x2, int nstep, void (*derivs)(double, SphVector[], SphVector[], int)) {
	double x, h;
	SphVector *v, *vout, *dv;

	v = (SphVector *)malloc(sizeof(SphVector) * nvar);
	vout = (SphVector *)malloc(sizeof(SphVector) * nvar);
	dv = (SphVector *)malloc(sizeof(SphVector) * nvar);

	for (int i = 0;i < nvar;i++) { 
		v[i] = vstart[i];
		y[i][0] = v[i]; 
	}

	xx[0] = x1;
	x = x1;
	h = (x2-x1)/nstep;

	for (int k = 0; k < nstep; k++) {
		(*derivs)(x, v, dv, nvar);
		rk4(v,dv,nvar,x,h,vout,derivs);
		if ((double)(x + h) == x) fprintf(stderr, "Step size too small in routine rkdumb");
		x += h;
		xx[k + 1] = x;
		for (int i = 0; i < nvar; i++) {
			v[i] = vout[i];
			y[i][k + 1] = v[i];
		}
	}

	free(dv);
	free(vout);
	free(v);
}

int main(int argc, char *argv[]) {
	int nvar = 2; //t, M
	int nstep;
	double endTime;
	SphVector vstart[1]; 
	FILE * output = fopen("output.txt", "w");
	double sd; 

	if(output == NULL) {
		printf("error opening file\n");
		return 0;
	}
	
	vstart[0].r = 500;
	vstart[0].theta = 0.01;
	vstart[0].phi = 0;

	Vector Happl = {0.0, 0.0, 2500.0};
	Vector Hanis = {0.0, 0.0, 0.0};

	//Get the step size for the simulation 
	if(argc < 2) {
		printf("Usage: %s [step size]\n", argv[0]);
		return 0;
	}
	endTime = strtof(argv[1], NULL);
	nstep = (int)ceil(endTime/TIMESTEP);
	sd = (3.45e-4)/sqrt(endTime);

	//Allocate memory for magnetization vector
	xx = (double *)malloc(sizeof(double) * (nstep + 1));
	//TODO: Address y row-major
	y = (SphVector **)malloc(sizeof(SphVector *) * nvar); 
	for(int i = 0; i < nvar; i++) {
		y[i] = (SphVector *)malloc(sizeof(SphVector) * (nstep + 1));
	}

	
	bool isDecreasing = true;
	for(int i = 0; i <= 200; i++) {
		//Thermal motion
		H.x = gaussian(0, sd);
		H.y = gaussian(0, sd);
		H.z = gaussian(0, sd);

		//Applied field
		H.x += Happl.x;
		H.y += Happl.y;
		H.z += Happl.z;

		//Anisotropy field
		anisotropyH(&Hanis, &vstart[0]);
		H.x += Hanis.x;
		H.y += Hanis.y;
		H.z += Hanis.z;

		//Simulate!
		rkdumb(vstart, nvar, 0.0, endTime, nstep, mDot); 

		fprintf(output, "%f\t%f\n", Happl.z, 500*cos(y[0][nstep].theta));
		#if 0
		for(int i = 0; i < (nstep + 1); i++) {
			if(!(i % 1000)) {
				fprintf(output, "%f\t%f\n", Happl.z, 500*cos(y[0][i].theta));
			}
		}
		#endif

		vstart[0].theta = y[0][nstep].theta;
		vstart[0].phi = y[0][nstep].phi;
		
		//Adjust applied field strength at endTime intervals	
		if(Happl.z + 2500.0 < 1.0) isDecreasing = false;
		isDecreasing ? (Happl.z -= 50.0) : (Happl.z += 50.0);
	}
	//Probably don't really need these
	free(xx);
	free(y);
	return 0;
}
