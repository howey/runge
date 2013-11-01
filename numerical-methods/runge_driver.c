#include <math.h>
#include <stdio.h>
#include "runge.h"

static double **y, *xx;

//Evaluates dthetadt[0..n-1] at t and theta[0..n-1]
//This particular derivative is not a function of t
void derivative(double t, double theta[], double dthetadt[], int n) {
	for(int i = 0; i < n; i++) {
		dthetadt[i] = (-2.2067e-12)*(pow(theta[i], 4) - 81e8);
	}
}
 
/*
Starting from initial values vstart[0..nvar-1] known at x1 use fourth-order Runge-Kutta
to advance nstep equal increments to x2. The user-supplied routine derivs(x,v,dvdx)
evaluates derivatives. Results are stored in the global variables y[0..nvar-1][0..nstep]
and xx[0..nstep].
*/
void rkdumb(double vstart[], int nvar, double x1, double x2, int nstep, void (*derivs)(double, double [], double [], int)) {
	//void rk4(double y[], double dydx[], int n, double x, double h, double yout[], void (*derivs)(double, double [], double []));
	int i,k;
	double x,h;
	double *v,*vout,*dv;

	v = (double *)malloc(sizeof(double) * nvar);
	vout = (double *)malloc(sizeof(double) * nvar);
	dv = (double *)malloc(sizeof(double) * nvar);

	for (i = 0;i < nvar;i++) { 
		v[i] = vstart[i];
		y[i][0] = v[i]; 
	}

	xx[0] = x1;
	x = x1;
	h = (x2-x1)/nstep;

	for (k = 0; k < nstep; k++) { 
		(*derivs)(x, v, dv, nvar);
		rk4(v,dv,nvar,x,h,vout,derivs);
		if ((double)(x+h) == x) fprintf(stderr, "Step size too small in routine rkdumb");
		x += h;
		xx[k+1] = x;
		for (i = 0;i < nvar;i++) {
			v[i] = vout[i];
			y[i][k+1] = v[i];
		}
	}

	free(dv);
	free(vout);
	free(v);
}

int main() {
	int nvar = 2;
	int nstep = 2;
	double vstart[1] = {1200};

	xx = (double *)malloc(sizeof(double) * (nstep + 1));
	//TODO: Address y row-major
	y = (double **)malloc(sizeof(double*) * nvar); 
	for(int i = 0; i < nvar; i++) {
		y[i] = (double *)malloc(sizeof(double) * (nstep + 1));
	}


	rkdumb(vstart, nvar, 0.0, 480.0, nstep, derivative); 
	for(int i = 0; i < (nstep + 1); i++) {
		printf("%f\n", y[0][i]);
	}
}
