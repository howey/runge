#include "nrutil.h"
#include <stdlib.h>

//Shamelessly copied from Numerical Recipes
/*
Given values for the variables y[1..n] and their derivatives dydx[1..n] known at x , use the
fourth-order Runge-Kutta method to advance the solution over an interval h and return the
incremented variables as yout[1..n] , which need not be a distinct array from y . The user
supplies the routine derivs(x,y,dydx) , which returns derivatives dydx at x .
*/
void rk4(double y[], double dydx[], int n, double x, double h, double yout[], void (*derivs)(double, double [], double [], int)) {
	int i;
	double xh, hh, h6, *dym, *dyt, *yt;

	#if 0	
	dym = vector(1, n);
	dyt = vector(1, n);
	yt = vector(1, n);
	#endif

	dym = (double *)malloc(sizeof(double) * n);
	dyt = (double *)malloc(sizeof(double) * n);
	yt = (double *)malloc(sizeof(double) * n);

	#if 0
	for(i = 0; i < n; i++) {
		dym[i] = 1;
		dyt[i] = 1;
		yt[i] = 1;
	}
	#endif

	hh = h * 0.5;
	h6 = h / 6.0;
	xh = x + hh;
	//First step
	for (i = 0; i < n; i++)
		yt[i] = y[i] + hh * dydx[i];
	//Second step
	(*derivs)(xh, yt, dyt, n);
	for (i = 0; i < n; i++)
		yt[i] = y[i] + hh * dyt[i];
	//Third step
	(*derivs)(xh, yt, dym, n);
	for (i = 0; i < n; i++) {
		yt[i] = y[i] + h * dym[i];
		dym[i] += dyt[i];
	}
	//Fourth step
	(*derivs)(x + h, yt, dyt, n);
	//Accumulate increments with proper weights
	for (i = 0; i < n; i++)
		yout[i] = y[i] + h6 * (dydx[i] + dyt[i] + 2.0 * dym[i]);
	
	free(yt);
	free(dyt);
	free(dym);
}
