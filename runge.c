#include "runge.h"

/* Time is in units of ns */
#ifndef RIGID
static const double ALPHA = 0.02; //dimensionless
static const double GAMMA = 1.76e-2; //(Oe*ns)^-1
static const double KANIS = 7.0e7; //erg*cm^-3
static const double TIMESTEP = (1e-7); //ns, the integrator timestep
static const double MSAT = 1100.0; //emu*cm^-3
static const double JEX = 1.1e-6; //erg*cm^-1
static const double ALEN = 3e-8; //cm
static const double TEMP = 300.0; //K
static const double BOLTZ = 1.38e-34; //g*cm^2*ns^-2*K^-1
static const double FIELDSTEP = 500.0; //Oe, the change in the applied field
static const double FIELDTIMESTEP = 0.1; //ns, time to wait before changing applied field
static const double FIELDRANGE = 130000.0; //Oe, create loop from FIELDRANGE to -FIELDRANGE Oe
#else
static const double ALPHA = 0.02; //dimensionless
static const double GAMMA = 1.76e-2; //(Oe*ns)^-1
static const double KANIS = 1.0e6; //erg*cm^-3
static const double TIMESTEP = (1e-4); //ns, the integrator timestep
static const double MSAT = 500.0; //emu*cm^-3
static const double JEX = 0; //erg*cm^-1
static const double ALEN = 1e-6; //cm
static const double TEMP = 300.0; //K
static const double BOLTZ = 1.38e-34; //g*cm^2*ns^-2*K^-1
static const double FIELDSTEP = 50.0; //Oe, the change in the applied field
static const double FIELDTIMESTEP = 1.0; //ns, time to wait before changing applied field
static const double FIELDRANGE = 4000.0; //Oe, create loop from FIELDRANGE to -FIELDRANGE Oe
#endif

//Computes the local applied field for every atom of moment M.
void computeField(Vector * H, const SphVector * M, Vector Happl, Vector * Htherm) {
	for(int i = 0; i < SIZE; i++) {
		//the applied field
		H[i].x = Happl.x;
		H[i].y = Happl.y;
		H[i].z = Happl.z;

		//the anisotropy field
		H[i].x += (1/M[i].r) * -2 * KANIS * cos(M[i].theta) * sin(M[i].theta) * cos(M[i].phi) * cos(M[i].theta);
		H[i].y += (1/M[i].r) * -2 * KANIS * cos(M[i].theta) * sin(M[i].theta) * sin(M[i].phi) * cos(M[i].theta);
		H[i].z += (1/M[i].r) * 2 * KANIS * cos(M[i].theta) * sin(M[i].theta) * sin(M[i].theta);

		//the field from random thermal motion
		H[i].x += Htherm[i].x;
		H[i].y += Htherm[i].y;
		H[i].z += Htherm[i].z;

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
		
		double Hex = 2.0 * JEX / (MSAT * ALEN * ALEN);

		H[i].x += Hex * (sin(up.theta) * cos(up.phi) + sin(down.theta) * cos(down.phi) + sin(left.theta) * cos(left.phi) + sin(right.theta) * cos(right.phi) + sin(front.theta) * cos(front.phi) + sin(back.theta) * cos(back.phi));
		H[i].y += Hex * (sin(up.theta) * sin(up.phi) + sin(down.theta) * sin(down.phi) + sin(left.theta) * sin(left.phi) + sin(right.theta) * sin(right.phi) + sin(front.theta) * sin(front.phi) + sin(back.theta) * sin(back.phi)); 
		H[i].z += Hex * (cos(up.theta) + cos(down.theta) + cos(left.theta) + cos(right.theta) + cos(front.theta) + cos(back.theta));
	}
}

//Shamelessly copied from Numerical Recipes
/*
Given values for the variables y[1..n] and their derivatives dydx[1..n] known at x , use the
fourth-order Runge-Kutta method to advance the solution over an interval h and return the
incremented variables as yout[1..n] , which need not be a distinct array from y . The user
supplies the routine derivs(x,y,dydx) , which returns derivatives dydx at x .
*/
void rk4(SphVector * y, SphVector * dydx, double h, SphVector * yout, Vector * H, Vector Happl, Vector * Htherm) {
	double hh, h6; 

	SphVector dym[SIZE];
	SphVector dyt[SIZE];
	SphVector yt[SIZE];

	//Scale field and time to avoid roundoff errors
	double scale = (2.0 * KANIS / MSAT);
	h *= scale;

	hh = h * 0.5;
	h6 = h / 6.0;

	//First step
	for (int i = 0; i < SIZE; i++) {
		//yt[i] = y[i] + hh * dydx[i];
		yt[i].r = y[i].r + hh * dydx[i].r;
		yt[i].phi = y[i].phi + hh * dydx[i].phi;
		yt[i].theta = y[i].theta + hh * dydx[i].theta;
	}
	//Second step
	computeField(H, yt, Happl, Htherm);	
	mDot(yt, dyt, H);
	for (int i = 0; i < SIZE; i++) {
		//yt[i] = y[i] + hh * dyt[i];
		yt[i].r = y[i].r + hh * dyt[i].r;
		yt[i].phi = y[i].phi + hh * dyt[i].phi;
		yt[i].theta = y[i].theta + hh * dyt[i].theta;
	}
	//Third step
	computeField(H, yt, Happl, Htherm);	
	mDot(yt, dym, H);
	for (int i = 0; i < SIZE; i++) {
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
	computeField(H, yt, Happl, Htherm);	
	mDot(yt, dyt, H);
	//Accumulate increments with proper weights
	for (int i = 0; i < SIZE; i++) {
		//yout[i] = y[i] + h6 * (dydx[i] + dyt[i] + 2.0 * dym[i]);
		yout[i].r = y[i].r + h6 * (dydx[i].r + dyt[i].r + 2.0 * dym[i].r);
		yout[i].phi = y[i].phi + h6 * (dydx[i].phi + dyt[i].phi + 2.0 * dym[i].phi);
		yout[i].theta = y[i].theta + h6 * (dydx[i].theta + dyt[i].theta + 2.0 * dym[i].theta);
	}
}

void mDot(SphVector M[], SphVector dMdt[], Vector H[]) {

	//Compute derivative
	for(int i = 0; i < SIZE; i++) {
		//Scale field to avoid roundoff errors
		double scale = (2.0 * KANIS / MSAT);
		Vector Hsc = H[i];
		Hsc.x /= scale;
		Hsc.y /= scale;
		Hsc.z /= scale;

		dMdt[i].r = 0;
		dMdt[i].phi = GAMMA * ((cos(M[i].theta) * sin(M[i].phi) * Hsc.y) / sin(M[i].theta) + (cos(M[i].theta) * cos(M[i].phi) * Hsc.x) / sin(M[i].theta) - Hsc.z) + ((ALPHA * GAMMA)/(1 + ALPHA * ALPHA)) * ((cos(M[i].phi) * Hsc.y) / sin(M[i].theta) - (sin(M[i].phi) * Hsc.x) / sin(M[i].theta));
		dMdt[i].theta = -GAMMA * (cos(M[i].phi) * Hsc.y - sin(M[i].phi) * Hsc.x) + ((ALPHA * GAMMA)/(1 + ALPHA * ALPHA)) * (cos(M[i].theta) * cos(M[i].phi) * Hsc.x - Hsc.z * sin(M[i].theta) + cos(M[i].theta) * sin(M[i].phi) * Hsc.y);
	}
}

/*
Starting from initial values vstart[0..nvar-1] known at x1 use fourth-order Runge-Kutta
to advance nstep equal increments to x2. The user-supplied routine derivs(x,v,dvdx)
evaluates derivatives. Results are stored in the global variables y[0..nvar-1][0..nstep]
and xx[0..nstep].
*/
void rkdumb(SphVector vstart[], double x1, double x2, int nstep, double * xx, SphVector ** y, Vector Happl) {
	double x, h;

	SphVector v[WIDTH * HEIGHT * DEPTH];
	SphVector vout[WIDTH * HEIGHT * DEPTH];
	SphVector dv[WIDTH * HEIGHT * DEPTH];
	Vector H[WIDTH * HEIGHT * DEPTH];
	Vector Htherm[WIDTH * HEIGHT * DEPTH];

	for (int i = 0; i < SIZE; i++) { 
		v[i] = vstart[i];
		y[i][0] = v[i]; 
	}

	xx[0] = x1;
	x = x1;
	h = (x2-x1)/nstep;

	for (int k = 0; k < nstep; k++) {

		for(int j = 0; j < SIZE; j++) {
			//the field from random thermal motion
			double vol = ALEN * ALEN * ALEN;
			double sd = (1e9) * sqrt((2 * BOLTZ * TEMP * ALPHA)/(GAMMA * vol * MSAT * TIMESTEP)); //time has units of s here

			double thermX = gaussian(0, sd); 
			double thermY = gaussian(0, sd); 
			double thermZ = gaussian(0, sd); 

			Htherm[j].x = thermX; 
			Htherm[j].y = thermY;
			Htherm[j].z = thermZ;
		}

		//Compute H field
		computeField(H, v, Happl, Htherm);	

		//Compute derivatives
		mDot(v, dv, H);
		
		rk4(v, dv, h, vout, H, Happl, Htherm);
		if ((double)(x + h) == x) fprintf(stderr, "Step size too small in routine rkdumb");
		x += h;
		xx[k + 1] = x;
		for (int i = 0; i < SIZE; i++) {
			v[i] = vout[i];
			y[i][k + 1] = v[i];
		}
	}
}

int main(void){
	int nstep;
	double endTime;
	SphVector vstart[SIZE]; 

	FILE * output = fopen("output.txt", "w");
	if(output == NULL) {
		printf("error opening file\n");
		return 1;
	}
	
	#if BENCHMARK
	FILE * times = fopen("times.txt", "w");
	if(times == NULL) {
		printf("error opening file: times.txt\n");
		return 1;
	}
	fprintf(times, "Time to simulate %fns\n", FIELDTIMESTEP);
	#endif

	//seed random number generator
	srand((unsigned int)time(NULL));

	for(int i = 0; i < SIZE; i++) {	
		vstart[i].r = MSAT;
		vstart[i].theta = 0.01;
		vstart[i].phi = 0;
	}

	double * xx;
	SphVector ** y;
	Vector Happl = {0.0, 0.0, FIELDRANGE}; 

	endTime = FIELDTIMESTEP; 
	endTime /= 100; //Reduce host memory usage
	nstep = ((int)ceil(endTime/TIMESTEP));

	xx = (double *)malloc(sizeof(double) * (long unsigned int)(nstep + 1));
	y = (SphVector **)malloc(sizeof(SphVector *) * (long unsigned int)SIZE); 
	for(int i = 0; i < SIZE; i++) {
		y[i] = (SphVector *)malloc(sizeof(SphVector) * (long unsigned int)(nstep + 1));
	}
	
	bool isDecreasing = true;
	for(int i = 0; i <= (4 * (int)(FIELDRANGE/FIELDSTEP)); i++) {
		#if BENCHMARK
		time_t start = time(NULL);
		#endif

		for(int j = 0; j < 100; j++) {
			//Simulate!
			rkdumb(vstart, endTime * j, endTime * (j + 1) - TIMESTEP, nstep, xx, y, Happl); 

			for(int k = 0; k < SIZE; k++) {
				vstart[k].r = y[k][nstep].r;
				vstart[k].theta = y[k][nstep].theta;
				vstart[k].phi = y[k][nstep].phi;
			}
		}

		#if BENCHMARK
		time_t end = time(NULL);
		fprintf(times, "%lds\n", (long)(end - start));
		fflush(times);
		#endif
	
		double mag = 0.0;	
		for(int k = 0; k < SIZE; k++) {
			mag += (y[k][nstep].r)*cos(y[k][nstep].theta);
		}

		mag /= (double)SIZE;
		fprintf(output, "%f\t%f\n", Happl.z, mag);
		fflush(output);

		//Adjust applied field strength at endTime intervals	
		if(Happl.z + FIELDRANGE < 1.0) isDecreasing = false;
		isDecreasing ? (Happl.z -= FIELDSTEP) : (Happl.z += FIELDSTEP);
	}

	//Probably don't really need these since we're about to exit the program
	free(xx);
	free(y);
	return 0;
}
