#include "runge.h"

/* Time is in units of ns */
static const double ALPHA = 0.02; //Dimensionless
static const double GAMMA = 1.76e-2; //(Oe*ns)^-1
static const double KANIS = 7.0e7; //erg*cm^-3
static const double TIMESTEP = (1e-5); //ns
static const double MSAT = 1100.0; //emu*cm^-3
static const double JEX = 1.1e-6; //erg*cm^-1
static const double ALEN = 3e-8; //cm
static const double TEMP = 300.0; //K
static const double BOLTZ = 1.38e-34; //g*cm^2*ns^-2*K^-1

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
		double vol = ALEN * ALEN * ALEN;
		double sd = (1e9) * sqrt((2 * BOLTZ * TEMP * ALPHA)/(GAMMA * vol * MSAT * TIMESTEP)); //time has units of s here

		double thermX = gaussian(0, sd); 
		double thermY = gaussian(0, sd); 
		double thermZ = gaussian(0, sd); 

		H[i].x += thermX;
		H[i].y += thermY;
		H[i].z += thermZ;

		//the exchange field
		SphVector up, down, left, right, front, back;

		if(i % (WIDTH * HEIGHT) < WIDTH)
			up = M[i + WIDTH * (HEIGHT - 1)]; 
		else
			up = M[i - WIDTH];

		if(i % (WIDTH * HEIGHT) > (WIDTH * (HEIGHT - 1) - 1))
			down = M[i - WIDTH * (HEIGHT - 1)];
		else
			down = M[i + WIDTH];	

		if(i % WIDTH == 0)
			left = M[i + (WIDTH - 1)];
		else
			left = M[i - 1];

		if((i + 1) % WIDTH == 0)
			right = M[i - (WIDTH - 1)];
		else
			right = M[i + 1];

		if(i < (WIDTH * HEIGHT))
			front = M[i + (WIDTH * HEIGHT * (DEPTH - 1))];
		else
			front = M[i - (WIDTH * HEIGHT)];

		if(i > (WIDTH * HEIGHT * (DEPTH - 1)))
			back = M[i - (WIDTH * HEIGHT * (DEPTH - 1))];
		else
			back = M[i + (WIDTH * HEIGHT)];
		
		double Hex = JEX / (MSAT * ALEN * ALEN);

		H[i].x += Hex * (sin(up.theta) * cos(up.phi) + sin(down.theta) * cos(down.phi) + sin(left.theta) * cos(left.phi) + sin(right.theta) * cos(right.phi) + sin(front.theta) * cos(front.phi) + sin(back.theta) * cos(back.phi));
		H[i].y += Hex * (sin(up.theta) * sin(up.phi) + sin(down.theta) * sin(down.phi) + sin(left.theta) * sin(left.phi) + sin(right.theta) * sin(right.phi) + sin(front.theta) * sin(front.phi) + sin(back.theta) * sin(back.phi)); 
		H[i].z += Hex * (cos(up.theta) + cos(down.theta) + cos(left.theta) + cos(right.theta) + cos(front.theta) + cos(back.theta));
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

	endTime = (1e9)*strtof(argv[1], NULL); //In ns

	endTime /= 100; //Reduce memory usage

	nstep = ((int)ceil(endTime/TIMESTEP));

	xx = (double *)malloc(sizeof(double) * (nstep + 1));
	//TODO: Address y row-major
	y = (SphVector **)malloc(sizeof(SphVector *) * nvar); 
	for(int i = 0; i < nvar; i++) {
		y[i] = (SphVector *)malloc(sizeof(SphVector) * (nstep + 1));
	}

	bool isDecreasing = true;
	Happl.x = 10.0;
	Happl.y = 10.0;
	Happl.z = 5000.0;

	for(int i = 0; i <= 400; i++) {
		for(int j = 0; j < 100; j++) {
			//Simulate!
			rkdumb(vstart, nvar, endTime * j, endTime * (j + 1) - TIMESTEP, nstep, mDot); 

			for(int i = 0; i < nvar; i++) {
				vstart[i].r = y[i][nstep].r;
				vstart[i].theta = y[i][nstep].theta;
				vstart[i].phi = y[i][nstep].phi;
			}
		}

		double mag = 0.0;
		for(int k = 0; k < nvar; k++) {
			//fprintf(output, "%f\t%f\n", Happl.z, (y[k][nstep].r)*cos(y[k][nstep].theta));
			mag += (y[k][nstep].r)*cos(y[k][nstep].theta);
		}
		mag /= (double)nvar;
		fprintf(output, "%f\t%f\n", Happl.z, mag);

		//Adjust applied field strength at endTime intervals	
		if(Happl.z + 5000.0 < 1.0) isDecreasing = false;
		isDecreasing ? (Happl.z -= 50.0) : (Happl.z += 50.0);
	}
	
	//Probably don't really need these since we're about to exit the program
	free(xx);
	free(y);
	return 0;
}
