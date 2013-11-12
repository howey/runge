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

/*
Starting from initial values vstart[0..nvar-1] known at x1 use fourth-order Runge-Kutta
to advance nstep equal increments to x2. The user-supplied routine derivs(x,v,dvdx)
evaluates derivatives. Results are stored in the global variables y[0..nvar-1][0..nstep]
and xx[0..nstep].
*/
void rkdumb(SphVector vstart[], int nvar, double x1, double x2, int nstep, void (*derivs)(double, SphVector[], SphVector[], int)) {
	double x, h;
	SphVector *v, *vout, *dv;
	Vector Hanis = {0.0, 0.0, 0.0};

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

		//Add in anisotropy
		anisotropyH(&Hanis, &y[0][k]);
		H.x += Hanis.x;
		H.y += Hanis.y;
		H.z += Hanis.z;

		(*derivs)(x, v, dv, nvar);
		rk4(v,dv,nvar,x,h,vout,derivs);
		if ((double)(x + h) == x) fprintf(stderr, "Step size too small in routine rkdumb");
		x += h;
		xx[k + 1] = x;
		for (int i = 0; i < nvar; i++) {
			v[i] = vout[i];
			y[i][k + 1] = v[i];
		}

		//Remove anisotropy for next step 
		anisotropyH(&Hanis, &y[0][k]);
		H.x -= Hanis.x;
		H.y -= Hanis.y;
		H.z -= Hanis.z;
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

		#if 0
		//Anisotropy field
		anisotropyH(&Hanis, &vstart[0]);
		H.x += Hanis.x;
		H.y += Hanis.y;
		H.z += Hanis.z;
		#endif

		//Simulate!
		rkdumb(vstart, nvar, 0.0, endTime, nstep, mDot); 

		fprintf(output, "%f\t%f\n", Happl.z, (y[0][nstep].r)*cos(y[0][nstep].theta));

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
