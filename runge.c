#include "runge.h"

/* Time is in units of ns */
static const double ALPHA = 0.02; 
static const double GAMMA = 1.76e-2;
static const double KANIS = 1e6;
static const double TIMESTEP = (1e-5);
static const double MSAT = 500;

static const double JEX = 1;

static double *xx;
static SphVector **y;
static Vector *H;
static Vector Happl;

//Shamelessly copied from Numerical Recipes
/*
Given values for the variables y[1..n] and their derivatives dydx[1..n] known at x , use the
fourth-order Runge-Kutta method to advance the solution over an interval h and return the
incremented variables as yout[1..n] , which need not be a distinct array from y . The user
supplies the routine derivs(x,y,dydx) , which returns derivatives dydx at x .
*/
void rk4(SphVector y[], SphVector dydx[], int n, double x, double h, SphVector yout[], void (*derivs)(double, SphVector[], SphVector[], int, Vector[])) {
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
	(*derivs)(xh, yt, dyt, n, H);
	for (int i = 0; i < n; i++) {
		//yt[i] = y[i] + hh * dyt[i];
		yt[i].r = y[i].r + hh * dyt[i].r;
		yt[i].phi = y[i].phi + hh * dyt[i].phi;
		yt[i].theta = y[i].theta + hh * dyt[i].theta;
	}
	//Third step
	(*derivs)(xh, yt, dym, n, H);
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
	(*derivs)(x + h, yt, dyt, n, H);
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

//Computes the local applied field for every atom of moment M. The global applied field is passed in as Happl. 
void computeField(Vector * H, Vector Happl, const SphVector * M, int nvar) {
	for(int i = 0; i < nvar; i++) {
		//the applied field
		H[i].x = Happl.x;
		H[i].y = Happl.y;
		H[i].z = Happl.z;

		//the anisotropy field
		H[i].x += (1/M[i].r) * -2 * KANIS * cos(M[i].theta) * sin(M[i].theta) * cos(M[i].phi) * cos(M[i].theta);
		H[i].y += (1/M[i].r) * -2 * KANIS * cos(M[i].theta) * sin(M[i].theta) * sin(M[i].phi) * cos(M[i].theta);
		H[i].z += (1/M[i].r) * 2 * KANIS * cos(M[i].theta) * sin(M[i].theta) * sin(M[i].theta);

		//the field from random thermal motion
		//TODO: sd doesn't have to be computed each time, it is constant
		#if USE_THERMAL
		double sd = 3.4e-4/sqrt(TIMESTEP * 1e-9);
		double thermX = gaussian(0, sd); 
		double thermY = gaussian(0, sd); 
		double thermZ = gaussian(0, sd); 

		H[i].x += thermX;
		H[i].y += thermY;
		H[i].z += thermZ;
		#endif

		//TODO: the exchange field
	}
}

void mDot(double t, SphVector M[], SphVector dMdt[], int nvar, Vector H[]) {

	//Compute derivative
	for(int i = 0; i < nvar; i++) {
		dMdt[i].r = 0;
		dMdt[i].phi = GAMMA * ((cos(M[i].theta) * sin(M[i].phi) * H[i].y) / sin(M[i].theta) + (cos(M[i].theta) * cos(M[i].phi) * H[i].x) / sin(M[i].theta) - H[i].z) + ((ALPHA * GAMMA)/(1 + ALPHA * ALPHA)) * ((cos(M[i].phi) * H[i].y) / sin(M[i].theta) - (sin(M[i].phi) * H[i].x) / sin(M[i].theta));
		dMdt[i].theta = -GAMMA * (cos(M[i].phi) * H[i].y - sin(M[i].phi) * H[i].x) + ((ALPHA * GAMMA)/(1 + ALPHA * ALPHA)) * (cos(M[i].theta) * cos(M[i].phi) * H[i].x - H[i].z * sin(M[i].theta) + cos(M[i].theta) * sin(M[i].phi) * H[i].y);
	}
}

/*
Starting from initial values vstart[0..nvar-1] known at x1 use fourth-order Runge-Kutta
to advance nstep equal increments to x2. The user-supplied routine derivs(x,v,dvdx)
evaluates derivatives. Results are stored in the global variables y[0..nvar-1][0..nstep]
and xx[0..nstep].
*/
void rkdumb(SphVector vstart[], int nvar, double x1, double x2, int nstep, void (*derivs)(double, SphVector[], SphVector[], int, Vector[])) {
	double x, h;
	SphVector *v, *vout, *dv;

	v = (SphVector *)malloc(sizeof(SphVector) * nvar);
	vout = (SphVector *)malloc(sizeof(SphVector) * nvar);
	dv = (SphVector *)malloc(sizeof(SphVector) * nvar);
	H = (Vector *)malloc(sizeof(Vector) * nvar);

	for (int i = 0;i < nvar;i++) { 
		v[i] = vstart[i];
		y[i][0] = v[i]; 
	}

	xx[0] = x1;
	x = x1;
	h = (x2-x1)/nstep;

	for (int k = 0; k < nstep; k++) {

		//Compute H field
		computeField(H, Happl, v, nvar);	

		//Compute derivatives
		(*derivs)(x, v, dv, nvar, H);
		
		rk4(v,dv,nvar,x,h,vout,derivs);
		if ((double)(x + h) == x) fprintf(stderr, "Step size too small in routine rkdumb");
		x += h;
		xx[k + 1] = x;
		for (int i = 0; i < nvar; i++) {
			v[i] = vout[i];
			y[i][k + 1] = v[i];
		}
	}

	free(H);
	free(dv);
	free(vout);
	free(v);
}

int main(int argc, char *argv[]) {
	int nvar = HEIGHT * WIDTH * DEPTH; //M for each particle 
	int nstep;
	double endTime;
	SphVector vstart[nvar]; 
	FILE * output = fopen("output.txt", "w");

	if(output == NULL) {
		printf("error opening file\n");
		return 0;
	}
	
	//seed random number generator
	srand(time(NULL));

	for(int i = 0; i < nvar; i++) {	
		vstart[i].r = MSAT;
		vstart[i].theta = 0.01;
		vstart[i].phi = 0;
	}


	//Get the step size for the simulation 
	if(argc < 2) {
		printf("Usage: %s [step size]\n", argv[0]);
		return 1;
	}

	//TODO: there has to be a better way of handling input
	endTime = (1e9)*strtof(argv[1], NULL); //In ns
	nstep = ((int)ceil(endTime/TIMESTEP));

	xx = (double *)malloc(sizeof(double) * (nstep + 1));
	//TODO: Address y row-major
	y = (SphVector **)malloc(sizeof(SphVector *) * nvar); 
	for(int i = 0; i < nvar; i++) {
		y[i] = (SphVector *)malloc(sizeof(SphVector) * (nstep + 1));
	}

	bool isDecreasing = true;
	Happl.x = 20.0;
	Happl.y = 20.0;
	Happl.z = 5000.0;

	for(int i = 0; i <= 400; i++) {
		//Simulate!
		rkdumb(vstart, nvar, 0.0, endTime, nstep, mDot); 

		for(int i = 0; i < nvar; i++) {
			vstart[i].r = y[i][nstep].r;
			vstart[i].theta = y[i][nstep].theta;
			vstart[i].phi = y[i][nstep].phi;
		}

		for(int k = 0; k < nvar; k++) {
			fprintf(output, "%f\t%f\n", Happl.z, (y[k][nstep].r)*cos(y[k][nstep].theta));
		}

		//Adjust applied field strength at endTime intervals	
		if(Happl.z + 5000.0 < 1.0) isDecreasing = false;
		isDecreasing ? (Happl.z -= 50.0) : (Happl.z += 50.0);
	}
	
	//Probably don't really need these since we're about to exit the program
	free(xx);
	free(y);
	return 0;
}
