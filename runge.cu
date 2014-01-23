#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <curand_kernel.h>
#include <time.h>
#include "vector.h"
#include "runge.h"
//#include "mars.h"


/* Time is in units of ns */
static const double ALPHA = 0.02; 
static const double GAMMA = 1.76e-2;
static const double KANIS = 1e6;
static const double TIMESTEP = (1e-5);
static const double MSAT = 500;

static const double JEX = 1;

static double *xx;
static SphVector **y;
static Vector H;
static Vector *H_d;
static SphVector *yout_d;

__global__ void rk4First(SphVector *yt_d, SphVector *y_d, SphVector * dydx_d, double hh, int n) {
	//TODO: Use shared memory
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if(i < n) {
		yt_d[i].r = y_d[i].r + hh * dydx_d[i].r;
		yt_d[i].phi = y_d[i].phi + hh * dydx_d[i].phi;
		yt_d[i].theta = y_d[i].theta + hh * dydx_d[i].theta;
	}
}

__global__ void rk4Second(SphVector *yt_d, SphVector *y_d, SphVector *dyt_d, double hh, int n) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if(i < n) {
		yt_d[i].r = y_d[i].r + hh * dyt_d[i].r;
		yt_d[i].phi = y_d[i].phi + hh * dyt_d[i].phi;
		yt_d[i].theta = y_d[i].theta + hh * dyt_d[i].theta;
	}
}

__global__ void rk4Third(SphVector *yt_d, SphVector *y_d, SphVector *dym_d, SphVector * dyt_d, double h, int n) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if(i < n) {
		yt_d[i].r = y_d[i].r + h * dym_d[i].r;
		dym_d[i].r += dyt_d[i].r;
		yt_d[i].phi = y_d[i].phi + h * dym_d[i].phi;
		dym_d[i].phi += dyt_d[i].phi;
		yt_d[i].theta = y_d[i].theta + h * dym_d[i].theta;
		dym_d[i].theta += dyt_d[i].theta;
	}
}

__global__ void rk4Fourth(SphVector *yout_d, SphVector *y_d, SphVector *dydx_d, SphVector *dyt_d, SphVector *dym_d, double h6, int n) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if(i < n) {
		yout_d[i].r = y_d[i].r + h6 * (dydx_d[i].r + dyt_d[i].r + 2.0 * dym_d[i].r);
		yout_d[i].phi = y_d[i].phi + h6 * (dydx_d[i].phi + dyt_d[i].phi + 2.0 * dym_d[i].phi);
		yout_d[i].theta = y_d[i].theta + h6 * (dydx_d[i].theta + dyt_d[i].theta + 2.0 * dym_d[i].theta);
	}
}

//Shamelessly copied from Numerical Recipes
/*
Given values for the variables y[1..n] and their derivatives dydx[1..n] known at x , use the
fourth-order Runge-Kutta method to advance the solution over an interval h and return the
incremented variables as yout[1..n] , which need not be a distinct array from y . The user
supplies the routine derivs(x,y,dydx) , which returns derivatives dydx at x .
*/
void rk4(SphVector y_d[], SphVector dydx_d[], int n, double x, double h, SphVector yout[], void (*derivs)(double, SphVector[], SphVector[], int, Vector[])) {
	double xh, hh, h6; 
	//SphVector *dym, *dyt, *yt;

	//device arrays
	//SphVector *dym_d, *dyt_d, *yt_d, *yout_d;
	SphVector *dym_d, *dyt_d, *yt_d;

	/*
	dym = (SphVector *)malloc(sizeof(SphVector) * n);
	dyt = (SphVector *)malloc(sizeof(SphVector) * n);
	yt = (SphVector *)malloc(sizeof(SphVector) * n);
	*/

	//allocate device arrays
	cudaMalloc((void **)&dym_d, sizeof(SphVector) * n);
	cudaMalloc((void **)&dyt_d, sizeof(SphVector) * n);
	cudaMalloc((void **)&yt_d, sizeof(SphVector) * n);
	//cudaMalloc((void **)&yout_d, sizeof(SphVector) * n);

	hh = h * 0.5;
	h6 = h / 6.0;
	xh = x + hh;

	//First step
	rk4First<<<ceil(n/512.0), 512>>>(yt_d, y_d, dydx_d, hh, n);
	/*
	for (int i = 0; i < n; i++) {
		//yt[i] = y[i] + hh * dydx[i];
		yt[i].r = y[i].r + hh * dydx[i].r;
		yt[i].phi = y[i].phi + hh * dydx[i].phi;
		yt[i].theta = y[i].theta + hh * dydx[i].theta;
	}
	*/
	//Second step
	(*derivs)<<<ceil(n/512.0), 512>>>(xh, yt_d, dyt_d, n, H_d);
	rk4Second<<<ceil(n/512.0), 512>>>(yt_d, y_d, dyt_d, hh, n);
	/*
	for (int i = 0; i < n; i++) {
		//yt[i] = y[i] + hh * dyt[i];
		yt[i].r = y[i].r + hh * dyt[i].r;
		yt[i].phi = y[i].phi + hh * dyt[i].phi;
		yt[i].theta = y[i].theta + hh * dyt[i].theta;
	}
	*/
	//Third step
	(*derivs)<<<ceil(n/512.0), 512>>>(xh, yt_d, dym_d, n, H_d);
	rk4Third<<<ceil(n/512.0), 512>>>(yt_d, y_d, dym_d, dyt_d, h, n);
	/*
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
	*/
	//Fourth step
	(*derivs)<<<ceil(n/512.0), 512>>>(x + h, yt_d, dyt_d, n, H_d);
	//Accumulate increments with proper weights
	rk4Fourth<<<ceil(n/512.0), 512>>>(yout_d, y_d, dydx_d, dyt_d, dym_d, h6, n);
	/*
	for (int i = 0; i < n; i++) {
		//yout[i] = y[i] + h6 * (dydx[i] + dyt[i] + 2.0 * dym[i]);
		yout[i].r = y[i].r + h6 * (dydx[i].r + dyt[i].r + 2.0 * dym[i].r);
		yout[i].phi = y[i].phi + h6 * (dydx[i].phi + dyt[i].phi + 2.0 * dym[i].phi);
		yout[i].theta = y[i].theta + h6 * (dydx[i].theta + dyt[i].theta + 2.0 * dym[i].theta);
	}
	*/

	//Copy yout to host
	cudaMemcpy(yout, yout_d, sizeof(SphVector) * n, cudaMemcpyDeviceToHost);
	
	/*
	free(yt);
	free(dyt);
	free(dym);
	*/

	//Free device arrays
	cudaFree(yt_d);
	cudaFree(dyt_d);
	cudaFree(dym_d);
	//cudaFree(yout_d);
}

/*
//Computes the anisotropy field and writes the result to a Vector H
void anisotropyH(Vector * Ha, const SphVector * M) {
	Ha->x = (1/M->r) * -2 * KANIS * cos(M->theta) * sin(M->theta) * cos(M->phi) * cos(M->theta);
	Ha->y = (1/M->r) * -2 * KANIS * cos(M->theta) * sin(M->theta) * sin(M->phi) * cos(M->theta);
	Ha->z = (1/M->r) * 2 * KANIS * cos(M->theta) * sin(M->theta) * sin(M->theta);
}
*/

//Computes the local applied field for every atom of moment M. The global applied field is passed in as H. 
__global__ void computeField(Vector * H_d, Vector H, SphVector * M, int nvar, unsigned long long seed) {
	//Thread coordinates
	int tx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
	int ty = blockIdx.y * BLOCK_SIZE + threadIdx.y;
	int tz = blockIdx.z * BLOCK_SIZE + threadIdx.z;
	int i = tz * WIDTH * HEIGHT +  ty * WIDTH + tx;
	
	curandState_t state;

	//initialize RNG
	//TODO: this should probably get a real seed
	curand_init(seed, i, 0, &state);

	if(i < nvar) {
		//the applied field
		H_d[i].x = H.x;
		H_d[i].y = H.y;
		H_d[i].z = H.z;

		//the anisotropy field
		H_d[i].x += (1/M[i].r) * -2 * KANIS * cos(M[i].theta) * sin(M[i].theta) * cos(M[i].phi) * cos(M[i].theta);
		H_d[i].y += (1/M[i].r) * -2 * KANIS * cos(M[i].theta) * sin(M[i].theta) * sin(M[i].phi) * cos(M[i].theta);
		H_d[i].z += (1/M[i].r) * 2 * KANIS * cos(M[i].theta) * sin(M[i].theta) * sin(M[i].theta);

		//the field from random thermal motion
		//TODO: sd doesn't have to be computed each time, it is constant
		#if USE_THERMAL
		double sd = 3.4e-4/sqrt(TIMESTEP * 1e-9);
		double thermX = sd * curand_normal_double(&state); 
		double thermY = sd * curand_normal_double(&state);
		double thermZ = sd * curand_normal_double(&state);

		H_d[i].x += thermX;
		H_d[i].y += thermY;
		H_d[i].z += thermZ;
		#endif

		//TODO: the exchange field
	}
}

__global__ void mDot(double t, SphVector M[], SphVector dMdt[], int nvar, Vector H[]) {
	int i = threadIdx.x + blockDim.x * blockIdx.x;

	//Compute derivative
	if(i < nvar) {
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
	dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
        dim3 gridDim(ceil(WIDTH/BLOCK_SIZE), ceil(HEIGHT/BLOCK_SIZE), ceil(DEPTH/BLOCK_SIZE));	
	SphVector *v, *vout, *dv;

	//device arrays
	SphVector *v_d, *dv_d;
	//Vector Hanis = {0.0, 0.0, 0.0};

	v = (SphVector *)malloc(sizeof(SphVector) * nvar);
	vout = (SphVector *)malloc(sizeof(SphVector) * nvar);
	dv = (SphVector *)malloc(sizeof(SphVector) * nvar);

	cudaMalloc((void **)&yout_d, sizeof(SphVector) * nvar);

	//allocate device memory for mDot
	cudaMalloc((void **)&v_d, sizeof(SphVector) * nvar);
	cudaMalloc((void **)&dv_d, sizeof(SphVector) * nvar);
	cudaMalloc((void **)&H_d, sizeof(SphVector) * nvar);
	
	for (int i = 0;i < nvar;i++) { 
		v[i] = vstart[i];
		y[i][0] = v[i]; 
	}

	xx[0] = x1;
	x = x1;
	h = (x2-x1)/nstep;

	double sd = 3.4e-4/sqrt(TIMESTEP * 1e-9);
	for (int k = 0; k < nstep; k++) {

		/*
		// Add in thermal motion
		#if USE_THERMAL 
		double thermX = gaussian(0, sd);
		double thermY = gaussian(0, sd);
		double thermZ = gaussian(0, sd);
		#endif

		//Add in anisotropy
		anisotropyH(&Hanis, &y[0][k]);
		H.x += Hanis.x;
		H.y += Hanis.y;
		H.z += Hanis.z;

		#if USE_THERMAL
		H.x += thermX;
		H.y += thermY;
		H.z += thermZ;
		#endif
		*/

		//Copy memory to device
		//After the first timestep, the value of v and yout_d are the same. d2d memcpy is much faster than h2s, so do it instead
		if(k == 0) cudaMemcpy(v_d, v, sizeof(SphVector) * nvar, cudaMemcpyHostToDevice);
		else cudaMemcpy(v_d, yout_d, sizeof(SphVector) * nvar, cudaMemcpyDeviceToDevice);

		//Launch kernel to compute H field
		computeField<<<gridDim, blockDim>>>(H_d, H, v_d, nvar, (unsigned long long)time(NULL));	

		//Launch kernel to compute derivatives
		(*derivs)<<<ceil(nvar/512.0), 512>>>(x, v_d, dv_d, nvar, H_d);
		
		rk4(v_d,dv_d,nvar,x,h,vout,derivs);
		if ((double)(x + h) == x) fprintf(stderr, "Step size too small in routine rkdumb");
		x += h;
		xx[k + 1] = x;
		for (int i = 0; i < nvar; i++) {
			v[i] = vout[i];
			y[i][k + 1] = v[i];
		}

		/*
		//Remove anisotropy for next step 
		anisotropyH(&Hanis, &y[0][k]);
		H.x -= Hanis.x;
		H.y -= Hanis.y;
		H.z -= Hanis.z;

		#if USE_THERMAL
		//Remove thermal motion
		H.x -= thermX;
		H.y -= thermY;
		H.z -= thermZ;
		#endif
		*/
	}

	free(dv);
	free(vout);
	free(v);
	cudaFree(yout_d);
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
	
	for(int i = 0; i < nvar; i++) {	
		vstart[i].r = MSAT;
		vstart[i].theta = 0.01;
		vstart[i].phi = 0;
	}

	Vector Happl = {20.0, 20.0, 5000.0};

	//Get the step size for the simulation 
	if(argc < 2) {
		printf("Usage: %s [step size]\n", argv[0]);
		return 1;
	}

	//TODO: there has to be a better way of handling input
	endTime = (1e9)*strtof(argv[1], NULL); //In ns
	nstep = ((int)ceil(endTime/TIMESTEP));

	nstep = ((nstep + 500) / 1000) * 1000; //round nstep to the nearest thousand

	if((nstep % 1000) == 0) nstep /= 1000; //Divide by 1000 to reduce host memory usage
	else {
		fprintf(stderr, "Error: please select step size such that nstep is a multiple of 1000\n");
		return 1;
	}

	xx = (double *)malloc(sizeof(double) * (nstep + 1));
	//TODO: Address y row-major
	y = (SphVector **)malloc(sizeof(SphVector *) * nvar); 
	for(int i = 0; i < nvar; i++) {
		y[i] = (SphVector *)malloc(sizeof(SphVector) * (nstep + 1));
	}

	
	bool isDecreasing = true;
	for(int i = 0; i <= 400; i++) {
		//Applied field
		H.x = Happl.x;
		H.y = Happl.y;
		H.z = Happl.z;

		for(int j = 0; j < 1000; j++) {
			//Simulate!
			rkdumb(vstart, nvar, j * endTime / 1000, j* endTime / 1000 + endTime - endTime/nstep, nstep, mDot); 

			for(int i = 0; i < nvar; i++) {
				vstart[i].r = y[i][nstep].r;
				vstart[i].theta = y[i][nstep].theta;
				vstart[i].phi = y[i][nstep].phi;
			}
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
