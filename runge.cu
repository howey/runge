#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <string.h>

#include <curand_kernel.h>
#include "runge.h"

//CUDA call error checking
//From https://stackoverflow.com/questions/14038589
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char * file, int line, bool abort=true) {
	if(code != cudaSuccess) {
		fprintf(stderr, "GPU Assert!: %s File: %s Line: %d\n", cudaGetErrorString(code), file, line);
		if(abort)
			exit(code);
	}
}

/* Time is in units of ns */
static const double ALPHA = 0.02; //dimensionless
static const double GAMMA = 1.76e-2; //(Oe*ns)^-1
static const double KANIS = 4.4e7; //erg*cm^-3
static const double TIMESTEP = (1e-7); //ns, the integrator timestep
static const double MSAT = 1100.0; //emu*cm^-3
static const double JEX = 1.1e-6; //erg*cm^-1
static const double ALEN = 3e-8; //cm
static const double TEMPAMB = 300.0; //K, the ambient temperature
static const double BOLTZ = 1.38e-34; //g*cm^2*ns^-2*K^-1
//static const double FIELDSTEP = 500.0; //Oe, the change in the applied field
//static const double FIELDTIMESTEP = 0.1; //ns, time to wait before changing applied field
//static const double FIELDRANGE = 130000.0; //Oe, create loop from FIELDRANGE to -FIELDRANGE Oe
static const double TPULSE = 1.269; //ns, the duration of the laser pulse
static const double TAU = 0.0551197; //ns, the time constant of the laser heating
static const double TEMPDELTA = 400.0; //K, the change in temperature produced by laser

static double *xx;
static SphVector **y;
static Vector H;
static Vector *H_d;
static Vector * Htherm_d;
static curandStateXORWOW_t *state;
static SphVector *dym_d, *dyt_d, *yt_d;
static SphVector *v_d, *dv_d;
static FILE * output;

__global__ void initializeRandom(curandStateXORWOW_t * state, int nvar, unsigned long long seed) {
	//the thread id
	int i = threadIdx.x + blockDim.x * blockIdx.x;

	//initialize RNG
	if(i < nvar)
		curand_init(seed, i, 0, &state[i]);
}
	SphVector y_s, dydx_s, yt_s;

		y_s = y_d[i];
		dydx_s = dydx_d[i];

		yt_s.r = y_s.r + hh * dydx_s.r;
		yt_s.phi = y_s.phi + hh * dydx_s.phi;
		yt_s.theta = y_s.theta + hh * dydx_s.theta;
	
		yt_d[i] = yt_s;

//y_d, a pointer to the state at iteration n
//H, the global applied field
	SphVector y_s, dyt_s, yt_s;

	   Since a halo element neighbors only one atom,
		y_s = y_d[i];
		dyt_s = dyt_d[i];

		yt_s.r = y_s.r + hh * dyt_s.r;
		yt_s.phi = y_s.phi + hh * dyt_s.phi;
		yt_s.theta = y_s.theta + hh * dyt_s.theta;

		yt_d[i] = yt_s;
	
	double hh, h6;
	int ix = threadIdx.x;
	SphVector y_s, dym_s, dyt_s, yt_s;

	int tx = blockIdx.x * BLOCK_SIZE + ix;
		y_s = y_d[i];
		dym_s = dym_d[i];
		dyt_s = dyt_d[i];

		yt_s.r = y_s.r + h * dym_s.r;
		dym_s.r += dyt_s.r;
		yt_s.phi = y_s.phi + h * dym_s.phi;
		dym_s.phi += dyt_s.phi;
		yt_s.theta = y_s.theta + h * dym_s.theta;
		dym_s.theta += dyt_s.theta;

		yt_d[i] = yt_s;
		dym_d[i] = dym_s;

	if(tx < WIDTH && ty < HEIGHT && tz < DEPTH) {
		//Load block into shared memory
	SphVector y_s, dydx_s, dyt_s, dym_s, yout_s;
		H_s[iz][iy][ix].y += (1/y_s[iz][iy][ix].r) * -2 * KANIS * cos(y_s[iz][iy][ix].theta) * sin(y_s[iz][iy][ix].theta) * sin(y_s[iz][iy][ix].phi) * cos(y_s[iz][iy][ix].theta);
		H_s[iz][iy][ix].z += (1/y_s[iz][iy][ix].r) * 2 * KANIS * cos(y_s[iz][iy][ix].theta) * sin(y_s[iz][iy][ix].theta) * sin(y_s[iz][iy][ix].theta);

		//the field from random thermal motion
		double vol = ALEN * ALEN * ALEN;
		double sd = (1e9) * sqrt((2 * BOLTZ * TEMP * ALPHA)/(GAMMA * vol * MSAT * TIMESTEP)); //time has units of s here

		double thermX = sd * curand_normal_double(&state[i]); 
		double thermY = sd * curand_normal_double(&state[i]);
		double thermZ = sd * curand_normal_double(&state[i]);

		H_s[iz][iy][ix].x += thermX;
		H_s[iz][iy][ix].y += thermY;
		H_s[iz][iy][ix].z += thermZ;


		//the exchange field
		SphVector up, down, left, right, front, back;

		//if(i % (WIDTH * HEIGHT) < WIDTH) //if at top of particle
		if(ty == 0)
			up = y_d[i + WIDTH * (HEIGHT - 1)]; 
		else if(iy > 0)
			up = y_s[iz][iy - 1][ix];
		else
			up = y_d[i - WIDTH];

		//if(i % (WIDTH * HEIGHT) > (WIDTH * (HEIGHT - 1) - 1)) //if at bottom of particle
		if(ty == (HEIGHT - 1))
			down = y_d[i - WIDTH * (HEIGHT - 1)];
		else if(iy < (blockDim.y - 1))
			down = y_s[iz][iy + 1][ix];
		else
			down = y_d[i + WIDTH];	

		//if(i % WIDTH == 0) //if at left
		if(tx == 0)
			left = y_d[i + (WIDTH - 1)]; 
		else if(ix > 0)
			left = y_s[iz][iy][ix - 1];
		else
			left = y_d[i - 1];

		//if((i + 1) % WIDTH == 0) //if at right
		if(tx == (WIDTH - 1))
			right = y_d[i - (WIDTH - 1)];
		else if(ix < (blockDim.x - 1))
			right = y_s[iz][iy][ix + 1];
		else
			right = y_d[i + 1];

		//if(i < (WIDTH * HEIGHT)) //if at front
		if(tz == 0)
			front = y_d[i + (WIDTH * HEIGHT * (DEPTH - 1))];
		else if(iz > 0)
			front = y_s[iz - 1][iy][ix];
		else
			front = y_d[i - (WIDTH * HEIGHT)];

		//if(i >= (WIDTH * HEIGHT * (DEPTH - 1))) //if at rear
		if(tz == (DEPTH - 1))
			back = y_d[i - (WIDTH * HEIGHT * (DEPTH - 1))];
		else if(iz < (blockDim.z - 1))
			back = y_s[iz + 1][iy][ix];
		else
			back = y_d[i + (WIDTH * HEIGHT)];

		double Hex = JEX / (MSAT * ALEN * ALEN);

		H_s[iz][iy][ix].x += Hex * (sin(up.theta) * cos(up.phi) + sin(down.theta) * cos(down.phi) + sin(left.theta) * cos(left.phi) + sin(right.theta) * cos(right.phi) + sin(front.theta) * cos(front.phi) + sin(back.theta) * cos(back.phi));
		H_s[iz][iy][ix].y += Hex * (sin(up.theta) * sin(up.phi) + sin(down.theta) * sin(down.phi) + sin(left.theta) * sin(left.phi) + sin(right.theta) * sin(right.phi) + sin(front.theta) * sin(front.phi) + sin(back.theta) * sin(back.phi)); 
		H_s[iz][iy][ix].z += Hex * (cos(up.theta) + cos(down.theta) + cos(left.theta) + cos(right.theta) + cos(front.theta) + cos(back.theta));

	}
	dydx_s[iz][iy][ix].phi = GAMMA * ((cos(y_s[iz][iy][ix].theta) * sin(y_s[iz][iy][ix].phi) * H_s[iz][iy][ix].y) / sin(y_s[iz][iy][ix].theta) + (cos(y_s[iz][iy][ix].theta) * cos(y_s[iz][iy][ix].phi) * H_s[iz][iy][ix].x) / sin(y_s[iz][iy][ix].theta) - H_s[iz][iy][ix].z) + ((ALPHA * GAMMA)/(1 + ALPHA * ALPHA)) * ((cos(y_s[iz][iy][ix].phi) * H_s[iz][iy][ix].y) / sin(y_s[iz][iy][ix].theta) - (sin(y_s[iz][iy][ix].phi) * H_s[iz][iy][ix].x) / sin(y_s[iz][iy][ix].theta));
	dydx_s[iz][iy][ix].theta = -GAMMA * (cos(y_s[iz][iy][ix].phi) * H_s[iz][iy][ix].y - sin(y_s[iz][iy][ix].phi) * H_s[iz][iy][ix].x) + ((ALPHA * GAMMA)/(1 + ALPHA * ALPHA)) * (cos(y_s[iz][iy][ix].theta) * cos(y_s[iz][iy][ix].phi) * H_s[iz][iy][ix].x - H_s[iz][iy][ix].z * sin(y_s[iz][iy][ix].theta) + cos(y_s[iz][iy][ix].theta) * sin(y_s[iz][iy][ix].phi) * H_s[iz][iy][ix].y);

	yt_d[iz][iy][ix].r = y_s[iz][iy][ix].r + hh * dydx_s[iz][iy][ix].r;
	yt_d[iz][iy][ix].phi = y_s[iz][iy][ix].phi + hh * dydx_s[iz][iy][ix].phi;
	yt_d[iz][iy][ix].theta = y_s[iz][iy][ix].theta + hh * dydx_s[iz][iy][ix].theta;
	dyt_d[iz][iy][ix].theta = -GAMMA * (cos(yt_d[iz][iy][ix].phi) * H_s[iz][iy][ix].y - sin(yt_d[iz][iy][ix].phi) * H_s[iz][iy][ix].x) + ((ALPHA * GAMMA)/(1 + ALPHA * ALPHA)) * (cos(yt_d[iz][iy][ix].theta) * cos(yt_d[iz][iy][ix].phi) * H_s[iz][iy][ix].x - H_s[iz][iy][ix].z * sin(yt_d[iz][iy][ix].theta) + cos(yt_d[iz][iy][ix].theta) * sin(yt_d[iz][iy][ix].phi) * H_s[iz][iy][ix].y);

	yt_d[iz][iy][ix].r = y_s[iz][iy][ix].r + hh * dyt_d[iz][iy][ix].r;
	yt_d[iz][iy][ix].phi = y_s[iz][iy][ix].phi + hh * dyt_d[iz][iy][ix].phi;
	yt_d[iz][iy][ix].theta = y_s[iz][iy][ix].theta + hh * dyt_d[iz][iy][ix].theta;
	dym_d[iz][iy][ix].theta = -GAMMA * (cos(yt_d[iz][iy][ix].phi) * H_s[iz][iy][ix].y - sin(yt_d[iz][iy][ix].phi) * H_s[iz][iy][ix].x) + ((ALPHA * GAMMA)/(1 + ALPHA * ALPHA)) * (cos(yt_d[iz][iy][ix].theta) * cos(yt_d[iz][iy][ix].phi) * H_s[iz][iy][ix].x - H_s[iz][iy][ix].z * sin(yt_d[iz][iy][ix].theta) + cos(yt_d[iz][iy][ix].theta) * sin(yt_d[iz][iy][ix].phi) * H_s[iz][iy][ix].y);

	if(i < n) {
		y_s = y_d[i];
		dydx_s = dydx_d[i];
		dyt_s = dyt_d[i];
		dym_s = dym_d[i];
	dym_d[iz][iy][ix].theta += dyt_d[iz][iy][ix].theta;
		
		yout_s.r = y_s.r + h6 * (dydx_s.r + dyt_s.r + 2.0 * dym_s.r);
		yout_s.phi = y_s.phi + h6 * (dydx_s.phi + dyt_s.phi + 2.0 * dym_s.phi);
		yout_s.theta = y_s.theta + h6 * (dydx_s.theta + dyt_s.theta + 2.0 * dym_s.theta);
	dyt_d[iz][iy][ix].theta = -GAMMA * (cos(yt_d[iz][iy][ix].phi) * H_s[iz][iy][ix].y - sin(yt_d[iz][iy][ix].phi) * H_s[iz][iy][ix].x) + ((ALPHA * GAMMA)/(1 + ALPHA * ALPHA)) * (cos(yt_d[iz][iy][ix].theta) * cos(yt_d[iz][iy][ix].phi) * H_s[iz][iy][ix].x - H_s[iz][iy][ix].z * sin(yt_d[iz][iy][ix].theta) + cos(yt_d[iz][iy][ix].theta) * sin(yt_d[iz][iy][ix].phi) * H_s[iz][iy][ix].y);

		yout_d[i] = yout_s;
	}
}

//Computes the local applied field for every atom of moment M. The global applied field is passed in as H, and the thermal motion as Htherm. 
__global__ void computeField(Vector * H_d, Vector H, Vector * Htherm_d, SphVector * M, int nvar) {
	/* Declare shared memory for CUDA block.
	   Since a halo element neighbors only one atom,
	   halo elements are not loaded into shared memory.
	   Instead, they are read from global memory as usual. */
	__shared__ SphVector M_s[BLOCK_SIZE][BLOCK_SIZE][BLOCK_SIZE];

	//Thread coordinates
	int tx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
	int ty = blockIdx.y * BLOCK_SIZE + threadIdx.y;
	int tz = blockIdx.z * BLOCK_SIZE + threadIdx.z;
	int i = tz * WIDTH * HEIGHT +  ty * WIDTH + tx;
	Vector H_t;

	if(tx < WIDTH && ty < HEIGHT && tz < DEPTH) {
		//Load block into shared memory
		M_s[threadIdx.z][threadIdx.y][threadIdx.x] = M[i];
		__syncthreads();

		//the applied field
		H_t.x = H.x;
		H_t.y = H.y;
		H_t.z = H.z;

		//the anisotropy field
		H_t.x += (1/M_s[threadIdx.z][threadIdx.y][threadIdx.x].r) * -2 * KANIS * cos(M_s[threadIdx.z][threadIdx.y][threadIdx.x].theta) * sin(M_s[threadIdx.z][threadIdx.y][threadIdx.x].theta) * cos(M_s[threadIdx.z][threadIdx.y][threadIdx.x].phi) * cos(M_s[threadIdx.z][threadIdx.y][threadIdx.x].theta);
		H_t.y += (1/M_s[threadIdx.z][threadIdx.y][threadIdx.x].r) * -2 * KANIS * cos(M_s[threadIdx.z][threadIdx.y][threadIdx.x].theta) * sin(M_s[threadIdx.z][threadIdx.y][threadIdx.x].theta) * sin(M_s[threadIdx.z][threadIdx.y][threadIdx.x].phi) * cos(M_s[threadIdx.z][threadIdx.y][threadIdx.x].theta);
		H_t.z += (1/M_s[threadIdx.z][threadIdx.y][threadIdx.x].r) * 2 * KANIS * cos(M_s[threadIdx.z][threadIdx.y][threadIdx.x].theta) * sin(M_s[threadIdx.z][threadIdx.y][threadIdx.x].theta) * sin(M_s[threadIdx.z][threadIdx.y][threadIdx.x].theta);

		//the field from random thermal motion
		Vector Htherm_s;
		Htherm_s = Htherm_d[i];

		H_t.x += Htherm_s.x;
		H_t.y += Htherm_s.y;
		H_t.z += Htherm_s.z;

		//the exchange field
		SphVector up, down, left, right, front, back;

		//if(i % (WIDTH * HEIGHT) < WIDTH) //if at top of particle
		if(ty == 0)
			up = M[i + WIDTH * (HEIGHT - 1)]; 
		else if(threadIdx.y > 0)
			up = M_s[threadIdx.z][threadIdx.y - 1][threadIdx.x];
		else
			up = M[i - WIDTH];

		//if(i % (WIDTH * HEIGHT) > (WIDTH * (HEIGHT - 1) - 1)) //if at bottom of particle
		if(ty == (HEIGHT - 1))
			down = M[i - WIDTH * (HEIGHT - 1)];
		else if(threadIdx.y < (blockDim.y - 1))
			down = M_s[threadIdx.z][threadIdx.y + 1][threadIdx.x];
		else
			down = M[i + WIDTH];	

		//if(i % WIDTH == 0) //if at left
		if(tx == 0)
			left = M[i + (WIDTH - 1)]; 
		else if(threadIdx.x > 0)
			left = M_s[threadIdx.z][threadIdx.y][threadIdx.x - 1];
		else
			left = M[i - 1];

		//if((i + 1) % WIDTH == 0) //if at right
		if(tx == (WIDTH - 1))
			right = M[i - (WIDTH - 1)];
		else if(threadIdx.x < (blockDim.x - 1))
			right = M_s[threadIdx.z][threadIdx.y][threadIdx.x + 1];
		else
			right = M[i + 1];

		//if(i < (WIDTH * HEIGHT)) //if at front
		if(tz == 0)
			front = M[i + (WIDTH * HEIGHT * (DEPTH - 1))];
		else if(threadIdx.z > 0)
			front = M_s[threadIdx.z - 1][threadIdx.y][threadIdx.x];
		else
			front = M[i - (WIDTH * HEIGHT)];

		//if(i >= (WIDTH * HEIGHT * (DEPTH - 1))) //if at rear
		if(tz == (DEPTH - 1))
			back = M[i - (WIDTH * HEIGHT * (DEPTH - 1))];
		else if(threadIdx.z < (blockDim.z - 1))
			back = M_s[threadIdx.z + 1][threadIdx.y][threadIdx.x];
		else
			back = M[i + (WIDTH * HEIGHT)];

		double Hex = 2.0 * JEX / (MSAT * ALEN * ALEN);

		H_t.x += Hex * (sin(up.theta) * cos(up.phi) + sin(down.theta) * cos(down.phi) + sin(left.theta) * cos(left.phi) + sin(right.theta) * cos(right.phi) + sin(front.theta) * cos(front.phi) + sin(back.theta) * cos(back.phi));
		H_t.y += Hex * (sin(up.theta) * sin(up.phi) + sin(down.theta) * sin(down.phi) + sin(left.theta) * sin(left.phi) + sin(right.theta) * sin(right.phi) + sin(front.theta) * sin(front.phi) + sin(back.theta) * sin(back.phi)); 
		H_t.z += Hex * (cos(up.theta) + cos(down.theta) + cos(left.theta) + cos(right.theta) + cos(front.theta) + cos(back.theta));

		H_d[i] = H_t;
	}
}

//Shamelessly copied from Numerical Recipes
/*
Given values for the variables y[1..n] and their derivatives dydx[1..n] known at x , use the
fourth-order Runge-Kutta method to advance the solution over an interval h and return the
incremented variables as yout[1..n] , which need not be a distinct array from y . The user
supplies the routine derivs(x,y,dydx) , which returns derivatives dydx at x .
*/
void rk4(SphVector y_d[], SphVector dydx_d[], int n, double x, double h, SphVector yout[], void (*derivs)(double, SphVector[], SphVector[], int, Vector[]), bool CopyToHost) {
	double xh, hh, h6; 
	dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
        dim3 gridDim(ceil((float)WIDTH/(float)BLOCK_SIZE), ceil((float)HEIGHT/(float)BLOCK_SIZE), ceil((float)DEPTH/(float)BLOCK_SIZE));	

	//device arrays

	//Scale field to avoid round-off errors
	h *= ((2.0 * KANIS) / MSAT);

	hh = h * 0.5;
	h6 = h / 6.0;
	xh = x + hh;

	//First step
	rk4First<<<ceil(n/512.0), 512>>>(yt_d, y_d, dydx_d, hh, n);

	//Second step
	computeField<<<gridDim, blockDim>>>(H_d, H, Htherm_d, yt_d, n); 
	(*derivs)<<<ceil(n/512.0), 512>>>(xh, yt_d, dyt_d, n, H_d);
	rk4Second<<<ceil(n/512.0), 512>>>(yt_d, y_d, dyt_d, hh, n);

	//Third step
	computeField<<<gridDim, blockDim>>>(H_d, H, Htherm_d, yt_d, n); 
	(*derivs)<<<ceil(n/512.0), 512>>>(xh, yt_d, dym_d, n, H_d);
	rk4Third<<<ceil(n/512.0), 512>>>(yt_d, y_d, dym_d, dyt_d, h, n);

	//Fourth step
	computeField<<<gridDim, blockDim>>>(H_d, H, Htherm_d, yt_d, n); 
	(*derivs)<<<ceil(n/512.0), 512>>>(x + h, yt_d, dyt_d, n, H_d);
	//Accumulate increments with proper weights
	rk4Fourth<<<ceil(n/512.0), 512>>>(yout_d, y_d, dydx_d, dyt_d, dym_d, h6, n);

	//Copy yout to host
	if(CopyToHost)
		cudaMemcpy(yout, yout_d, sizeof(SphVector) * n, cudaMemcpyDeviceToHost);
	
	//cudaFree(yout_d);
}

__global__ void computeHtherm(Vector * Htherm_d, int nvar, curandStateXORWOW_t * state, double temp) {
		int i = threadIdx.x + blockDim.x * blockIdx.x;

		if(i < nvar) {
			//the field from random thermal motion
			double vol = ALEN * ALEN * ALEN;
			double sd = (1e9) * sqrt((2 * BOLTZ * temp * ALPHA)/(GAMMA * vol * MSAT * TIMESTEP)); //time has units of s here

			double thermX = sd * curand_normal_double(&state[i]); 
			double thermY = sd * curand_normal_double(&state[i]);
			double thermZ = sd * curand_normal_double(&state[i]);

			Vector Htherm_s;

			Htherm_s.x = thermX;
			Htherm_s.y = thermY;
			Htherm_s.z = thermZ;
		
			Htherm_d[i] = Htherm_s;
		}
}

__global__ void mDot(double t, SphVector M[], SphVector dMdt[], int nvar, Vector H[]) {
	int i = threadIdx.x + blockDim.x * blockIdx.x;

	//Compute derivative
	if(i < nvar) {
		SphVector M_s = M[i], dMdt_s;
		Vector H_s = H[i];

		//Scale field to avoid round-off errors
		H_s.x /= ((2.0 * KANIS) / MSAT);
		H_s.y /= ((2.0 * KANIS) / MSAT);
		H_s.z /= ((2.0 * KANIS) / MSAT);

		dMdt_s.r = 0;
		dMdt_s.phi = GAMMA * ((cos(M_s.theta) * sin(M_s.phi) * H_s.y) / sin(M_s.theta) + (cos(M_s.theta) * cos(M_s.phi) * H_s.x) / sin(M_s.theta) - H_s.z) + ((ALPHA * GAMMA)/(1 + ALPHA * ALPHA)) * ((cos(M_s.phi) * H_s.y) / sin(M_s.theta) - (sin(M_s.phi) * H_s.x) / sin(M_s.theta));
		dMdt_s.theta = -GAMMA * (cos(M_s.phi) * H_s.y - sin(M_s.phi) * H_s.x) + ((ALPHA * GAMMA)/(1 + ALPHA * ALPHA)) * (cos(M_s.theta) * cos(M_s.phi) * H_s.x - H_s.z * sin(M_s.theta) + cos(M_s.theta) * sin(M_s.phi) * H_s.y);
	
		dMdt[i] = dMdt_s;
	}
}

/*
Starting from initial values vstart[0..nvar-1] known at x1 use fourth-order Runge-Kutta
to advance nstep equal increments to x2. The user-supplied routine derivs(x,v,dvdx)
evaluates derivatives. Results are stored in the global variables y[0..nvar-1][0..nstep]
and xx[0..nstep].
*/
void rkdumb(SphVector vstart[], int nvar, double x1, double x2, int nstep, void (*derivs)(double, SphVector[], SphVector[], int, Vector[])) {
	double x, h, temp = TEMPAMB;
	dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
        dim3 gridDim(ceil((float)WIDTH/(float)BLOCK_SIZE), ceil((float)HEIGHT/(float)BLOCK_SIZE), ceil((float)DEPTH/(float)BLOCK_SIZE));	
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

		if(!(k % 1000)) {
			double t = x1 + k * TIMESTEP;
			if(t <= TPULSE) {
				double g = 1 - exp(-t/TAU);
				temp = TEMPAMB + TEMPDELTA * g;
			}
			else if(t > TPULSE) {
				double g = exp(-(t - TPULSE)/TAU);
				temp = TEMPAMB + TEMPDELTA * g;
			}
		}

		if(k == 0){
			fprintf(output, "%f\t", temp);
		}

		//Copy memory to device
		//After the first timestep, the value of v and yout_d are the same. d2d memcpy is much faster than h2s, so do it instead
		if(k == 0) {
			gpuErrchk( cudaMemcpy(v_d, v, sizeof(SphVector) * nvar, cudaMemcpyHostToDevice) );
		}
		else {
			SphVector *t_d = v_d;
			v_d = yout_d;
			yout_d = t_d;
		}

		//Generate thermal noise
		computeHtherm<<<ceil(nvar/512.0), 512>>>(Htherm_d, nvar, state, temp);

		//Launch kernel to compute H field
		computeField<<<gridDim, blockDim>>>(H_d, H, Htherm_d, v_d, nvar); 

		rk4Kernel<<<gridDim, blockDim>>>(v_d, nvar, x, h, yout_d, H, state);
		gpuErrchk( cudaPeekAtLastError() );
		gpuErrchk( cudaDeviceSynchronize() );
		if(k == (nstep - 1)) {
			gpuErrchk( cudaMemcpy(vout, yout_d, sizeof(SphVector) * nvar, cudaMemcpyDeviceToHost) );
		}
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
	int nvar = HEIGHT * WIDTH * DEPTH; //M for each particle 
	int nstep;
	double endTime;
	SphVector vstart[nvar]; 

	char fileName[128] = {0};
	char * jobId = getenv("PBS_JOBID");
	strcpy(fileName, jobId);

	//Check if a file named fileName already exists, needed for two-GPU jobs
	struct stat buffer;
	int fd = stat(fileName, &buffer);
	if(fd < 0) {
		//file doesn't exist
		output = fopen(fileName, "w");
	}
	else {
		//file exists
		strcat(fileName, "-1");
		output = fopen(fileName, "w");
	}

	#if BENCHMARK
	FILE * times = fopen("times.txt", "w");
	if(times == NULL) {
		printf("error opening file: times.txt\n");
		return 1;
	}
	//fprintf(times, "Time to simulate %fns\n", FIELDTIMESTEP);
	#endif

	//Initialize random number generator
	unsigned long long seed = time(NULL);
	cudaMalloc((void **)&state, sizeof(curandStateXORWOW_t) * nvar);
	initializeRandom<<<ceil(nvar/512.0), 512>>>(state, nvar, seed);
	
	//allocate device arrays
	cudaMalloc((void **)&dym_d, sizeof(SphVector) * nvar);
	cudaMalloc((void **)&dyt_d, sizeof(SphVector) * nvar);
	cudaMalloc((void **)&yt_d, sizeof(SphVector) * nvar);
	cudaMalloc((void **)&yout_d, sizeof(SphVector) * nvar);

	//allocate device memory for mDot
	cudaMalloc((void **)&v_d, sizeof(SphVector) * nvar);
	cudaMalloc((void **)&dv_d, sizeof(SphVector) * nvar);
	cudaMalloc((void **)&H_d, sizeof(Vector) * nvar);

	//allocate device memory for thermal motion
	cudaMalloc((void **)&Htherm_d, sizeof(Vector) * nvar);

	for(int i = 0; i < nvar; i++) {	
		vstart[i].r = MSAT;
		vstart[i].theta = 3.13;
		vstart[i].phi = 0;
	}

	//Vector Happl = {0.0, 0.0, FIELDRANGE};
	Vector Happl = {0.0, 0.0, 0.5e4};	
	//endTime = FIELDTIMESTEP; 
	//endTime = 2.538;
	endTime = 2.0 * TPULSE;
	endTime /= 100; //Reduce host memory usage
	nstep = ((int)ceil(endTime/TIMESTEP));

	xx = (double *)malloc(sizeof(double) * (nstep + 1));
	y = (SphVector **)malloc(sizeof(SphVector *) * nvar); 
	for(int i = 0; i < nvar; i++) {
		y[i] = (SphVector *)malloc(sizeof(SphVector) * (nstep + 1));
	}

	//Applied field
	H.x = Happl.x;
	H.y = Happl.y;
	H.z = Happl.z;

	#if BENCHMARK
	time_t start = time(NULL);
	#endif

	for(int j = 0; j < 100; j++) {
		//Simulate!
		rkdumb(vstart, nvar, endTime * j, endTime * (j + 1) - TIMESTEP, nstep, mDot); 

		for(int i = 0; i < nvar; i++) {
			vstart[i].r = y[i][nstep].r;
			vstart[i].theta = y[i][nstep].theta;
			vstart[i].phi = y[i][nstep].phi;
		}

		double mag = 0.0;	
		for(int k = 0; k < nvar; k++) {
			mag += (y[k][nstep].r)*cos(y[k][nstep].theta);
		}

		mag /= (double)nvar;
		fprintf(output, "%f\t%f\n", endTime * (j + 1), mag);
		fflush(output);
	}

	#if BENCHMARK
	time_t end = time(NULL);
	fprintf(times, "%lds\n", (long)(end - start));
	fflush(times);
	#endif

	//Probably don't really need these since we're about to exit the program
	free(xx);
	free(y);
	cudaFree(state);

	//Free device arrays
	cudaFree(yt_d);
	cudaFree(dyt_d);
	cudaFree(dym_d);
	cudaFree(v_d);
	cudaFree(dv_d);
	cudaFree(H_d);
	cudaFree(Htherm_d);
	cudaFree(yout_d);

	return 0;
}
