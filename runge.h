
//Shamelessly copied from Numerical Recipes
/*
Given values for the variables y[1..n] and their derivatives dydx[1..n] known at x , use the
fourth-order Runge-Kutta method to advance the solution over an interval h and return the
incremented variables as yout[1..n] , which need not be a distinct array from y . The user
supplies the routine derivs(x,y,dydx) , which returns derivatives dydx at x .
*/
void rk4(SphVector y[], SphVector dydx[], int n, double x, double h, SphVector yout[], void (*derivs)(double, SphVector[], SphVector[], int)) {
	double xh, hh, h6; 
	SphVector *dym, *dyt, *yt;

	dym = (SphVector *)malloc(sizeof(SphVector) * n);
	dyt = (SphVector *)malloc(sizeof(SphVector) * n);
	yt = (SphVector *)malloc(sizeof(SphVector) * n);

	hh = h * 0.5;
	h6 = h / 6.0;
	xh = x + hh;

	//First step
	for (int i = 0; i < n; i++) {
		//yt[i] = y[i] + hh * dydx[i];
		yt[i].r = y[i].r + hh * dydx[i].r;
		yt[i].phi = y[i].phi + hh * dydx[i].phi;
		yt[i].theta = y[i].theta + hh * dydx[i].theta;
	}
	//Second step
	(*derivs)(xh, yt, dyt, n);
	for (int i = 0; i < n; i++) {
		//yt[i] = y[i] + hh * dyt[i];
		yt[i].r = y[i].r + hh * dyt[i].r;
		yt[i].phi = y[i].phi + hh * dyt[i].phi;
		yt[i].theta = y[i].theta + hh * dyt[i].theta;
	}
	//Third step
	(*derivs)(xh, yt, dym, n);
	for (int i = 0; i < n; i++) {
		//yt[i] = y[i] + h * dym[i];
		//dym[i] += dyt[i];
		yt[i].r = y[i].r + h * dym[i].r;
		dym[i].r += dyt[i].r;
		yt[i].phi = y[i].phi + h * dym[i].phi;
		dym[i].phi += dyt[i].phi;
		yt[i].theta = y[i].theta + h * dym[i].theta;
		dym[i].theta += dyt[i].theta;
	}
	//Fourth step
	(*derivs)(x + h, yt, dyt, n);
	//Accumulate increments with proper weights
	for (int i = 0; i < n; i++) {
		//yout[i] = y[i] + h6 * (dydx[i] + dyt[i] + 2.0 * dym[i]);
		yout[i].r = y[i].r + h6 * (dydx[i].r + dyt[i].r + 2.0 * dym[i].r);
		yout[i].phi = y[i].phi + h6 * (dydx[i].phi + dyt[i].phi + 2.0 * dym[i].phi);
		yout[i].theta = y[i].theta + h6 * (dydx[i].theta + dyt[i].theta + 2.0 * dym[i].theta);
	}
	free(yt);
	free(dyt);
	free(dym);
}
