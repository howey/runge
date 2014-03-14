#include <curand_kernel.h>
#include "runge.h"

/* Time is in units of ns */
static const double ALPHA = 0.02; //dimensionless
static const double GAMMA = 1.76e-2; //(Oe*ns)^-1
static const double KANIS = 7.0e7; //erg*cm^-3
static const double TIMESTEP = (1e-7); //ns, the integrator timestep
static const double MSAT = 1100.0; //emu*cm^-3
static const double JEX = 1.1e-6; //erg*cm^-1
static const double ALEN = 3e-8; //cm
static const double TEMP = 0.0; //K
static const double BOLTZ = 1.38e-34; //g*cm^2*ns^-2*K^-1
static const double FIELDSTEP = 500.0; //Oe, the change in the applied field
static const double FIELDTIMESTEP = 0.1; //ns, time to wait before changing applied field
static const double FIELDRANGE = 130000.0; //Oe, create loop from FIELDRANGE to -FIELDRANGE Oe

static double *xx;
static SphVector **y;
static Vector H;
static Vector *H_d;
static curandStateXORWOW_t *state;

__global__ void initializeRandom(curandStateXORWOW_t * state, int nvar, unsigned long long seed) {
	//the thread id
	int i = threadIdx.x + blockDim.x * blockIdx.x;

	//initialize RNG
	if(i < nvar)
		curand_init(seed, i, 0, &state[i]);
}

__global__ void rk4Kernel(SphVector * y_d, SphVector * dydx_d, int n, double x, double h, SphVector * yout_d, Vector * H) {
	//device arrays
	__shared__ SphVector dym_d[VECTOR_SIZE];
	__shared__ SphVector dyt_d[VECTOR_SIZE];
	__shared__ SphVector yt_d[VECTOR_SIZE];
	__shared__ Vector H_s[VECTOR_SIZE];
	__shared__ SphVector y_s[VECTOR_SIZE];
	__shared__ SphVector dydx_s[VECTOR_SIZE];

	double hh, h6;
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int tx = threadIdx.x;


	hh = h * 0.5;
	h6 = h / 6.0;

	//Load field into shared memory
	if(i < n) {
		H_s[tx] = H[i];
	}

	//Load input into shared memory
	if(i < n) {
		y_s[tx] = y_d[i];
	}

	//First step
	dydx_s[tx].r = 0;
	dydx_s[tx].phi = GAMMA * ((cos(y_s[tx].theta) * sin(y_s[tx].phi) * H_s[tx].y) / sin(y_s[tx].theta) + (cos(y_s[tx].theta) * cos(y_s[tx].phi) * H_s[tx].x) / sin(y_s[tx].theta) - H_s[tx].z) + ((ALPHA * GAMMA)/(1 + ALPHA * ALPHA)) * ((cos(y_s[tx].phi) * H_s[tx].y) / sin(y_s[tx].theta) - (sin(y_s[tx].phi) * H_s[tx].x) / sin(y_s[tx].theta));
	dydx_s[tx].theta = -GAMMA * (cos(y_s[tx].phi) * H_s[tx].y - sin(y_s[tx].phi) * H_s[tx].x) + ((ALPHA * GAMMA)/(1 + ALPHA * ALPHA)) * (cos(y_s[tx].theta) * cos(y_s[tx].phi) * H_s[tx].x - H_s[tx].z * sin(y_s[tx].theta) + cos(y_s[tx].theta) * sin(y_s[tx].phi) * H_s[tx].y);

	yt_d[tx].r = y_s[tx].r + hh * dydx_s[tx].r;
	yt_d[tx].phi = y_s[tx].phi + hh * dydx_s[tx].phi;
	yt_d[tx].theta = y_s[tx].theta + hh * dydx_s[tx].theta;

	//Second step
	dyt_d[tx].r = 0;
	dyt_d[tx].phi = GAMMA * ((cos(yt_d[tx].theta) * sin(yt_d[tx].phi) * H_s[tx].y) / sin(yt_d[tx].theta) + (cos(yt_d[tx].theta) * cos(yt_d[tx].phi) * H_s[tx].x) / sin(yt_d[tx].theta) - H_s[tx].z) + ((ALPHA * GAMMA)/(1 + ALPHA * ALPHA)) * ((cos(yt_d[tx].phi) * H_s[tx].y) / sin(yt_d[tx].theta) - (sin(yt_d[tx].phi) * H_s[tx].x) / sin(yt_d[tx].theta));
	dyt_d[tx].theta = -GAMMA * (cos(yt_d[tx].phi) * H_s[tx].y - sin(yt_d[tx].phi) * H_s[tx].x) + ((ALPHA * GAMMA)/(1 + ALPHA * ALPHA)) * (cos(yt_d[tx].theta) * cos(yt_d[tx].phi) * H_s[tx].x - H_s[tx].z * sin(yt_d[tx].theta) + cos(yt_d[tx].theta) * sin(yt_d[tx].phi) * H_s[tx].y);

	yt_d[tx].r = y_s[tx].r + hh * dyt_d[tx].r;
	yt_d[tx].phi = y_s[tx].phi + hh * dyt_d[tx].phi;
	yt_d[tx].theta = y_s[tx].theta + hh * dyt_d[tx].theta;

	//Third step
	dym_d[tx].r = 0;
	dym_d[tx].phi = GAMMA * ((cos(yt_d[tx].theta) * sin(yt_d[tx].phi) * H_s[tx].y) / sin(yt_d[tx].theta) + (cos(yt_d[tx].theta) * cos(yt_d[tx].phi) * H_s[tx].x) / sin(yt_d[tx].theta) - H_s[tx].z) + ((ALPHA * GAMMA)/(1 + ALPHA * ALPHA)) * ((cos(yt_d[tx].phi) * H_s[tx].y) / sin(yt_d[tx].theta) - (sin(yt_d[tx].phi) * H_s[tx].x) / sin(yt_d[tx].theta));
	dym_d[tx].theta = -GAMMA * (cos(yt_d[tx].phi) * H_s[tx].y - sin(yt_d[tx].phi) * H_s[tx].x) + ((ALPHA * GAMMA)/(1 + ALPHA * ALPHA)) * (cos(yt_d[tx].theta) * cos(yt_d[tx].phi) * H_s[tx].x - H_s[tx].z * sin(yt_d[tx].theta) + cos(yt_d[tx].theta) * sin(yt_d[tx].phi) * H_s[tx].y);

	yt_d[tx].r = y_s[tx].r + h * dym_d[tx].r;
	dym_d[tx].r += dyt_d[tx].r;
	yt_d[tx].phi = y_s[tx].phi + h * dym_d[tx].phi;
	dym_d[tx].phi += dyt_d[tx].phi;
	yt_d[tx].theta = y_s[tx].theta + h * dym_d[tx].theta;
	dym_d[tx].theta += dyt_d[tx].theta;

	//Fourth step
	dyt_d[tx].r = 0;
	dyt_d[tx].phi = GAMMA * ((cos(yt_d[tx].theta) * sin(yt_d[tx].phi) * H_s[tx].y) / sin(yt_d[tx].theta) + (cos(yt_d[tx].theta) * cos(yt_d[tx].phi) * H_s[tx].x) / sin(yt_d[tx].theta) - H_s[tx].z) + ((ALPHA * GAMMA)/(1 + ALPHA * ALPHA)) * ((cos(yt_d[tx].phi) * H_s[tx].y) / sin(yt_d[tx].theta) - (sin(yt_d[tx].phi) * H_s[tx].x) / sin(yt_d[tx].theta));
	dyt_d[tx].theta = -GAMMA * (cos(yt_d[tx].phi) * H_s[tx].y - sin(yt_d[tx].phi) * H_s[tx].x) + ((ALPHA * GAMMA)/(1 + ALPHA * ALPHA)) * (cos(yt_d[tx].theta) * cos(yt_d[tx].phi) * H_s[tx].x - H_s[tx].z * sin(yt_d[tx].theta) + cos(yt_d[tx].theta) * sin(yt_d[tx].phi) * H_s[tx].y);

	if(i < n) {
		yout_d[i].r = y_s[tx].r + h6 * (dydx_d[i].r + dyt_d[tx].r + 2.0 * dym_d[tx].r);
		yout_d[i].phi = y_s[tx].phi + h6 * (dydx_d[i].phi + dyt_d[tx].phi + 2.0 * dym_d[tx].phi);
		yout_d[i].theta = y_s[tx].theta + h6 * (dydx_d[i].theta + dyt_d[tx].theta + 2.0 * dym_d[tx].theta);
	}
}

//Computes the local applied field for every atom of moment M. The global applied field is passed in as H. 
__global__ void computeField(Vector * H_d, Vector H, SphVector * M, int nvar, curandStateXORWOW_t * state) {
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

	if(tx < WIDTH && ty < HEIGHT && tz < DEPTH) {
		//Load block into shared memory
		M_s[threadIdx.z][threadIdx.y][threadIdx.x] = M[i];
		__syncthreads();

		//the applied field
		H_d[i].x = H.x;
		H_d[i].y = H.y;
		H_d[i].z = H.z;

		//the anisotropy field
		H_d[i].x += (1/M_s[threadIdx.z][threadIdx.y][threadIdx.x].r) * -2 * KANIS * cos(M_s[threadIdx.z][threadIdx.y][threadIdx.x].theta) * sin(M_s[threadIdx.z][threadIdx.y][threadIdx.x].theta) * cos(M_s[threadIdx.z][threadIdx.y][threadIdx.x].phi) * cos(M_s[threadIdx.z][threadIdx.y][threadIdx.x].theta);
		H_d[i].y += (1/M_s[threadIdx.z][threadIdx.y][threadIdx.x].r) * -2 * KANIS * cos(M_s[threadIdx.z][threadIdx.y][threadIdx.x].theta) * sin(M_s[threadIdx.z][threadIdx.y][threadIdx.x].theta) * sin(M_s[threadIdx.z][threadIdx.y][threadIdx.x].phi) * cos(M_s[threadIdx.z][threadIdx.y][threadIdx.x].theta);
		H_d[i].z += (1/M_s[threadIdx.z][threadIdx.y][threadIdx.x].r) * 2 * KANIS * cos(M_s[threadIdx.z][threadIdx.y][threadIdx.x].theta) * sin(M_s[threadIdx.z][threadIdx.y][threadIdx.x].theta) * sin(M_s[threadIdx.z][threadIdx.y][threadIdx.x].theta);

		//the field from random thermal motion
		double vol = ALEN * ALEN * ALEN;
		double sd = (1e9) * sqrt((2 * BOLTZ * TEMP * ALPHA)/(GAMMA * vol * MSAT * TIMESTEP)); //time has units of s here

		double thermX = sd * curand_normal_double(&state[i]); 
		double thermY = sd * curand_normal_double(&state[i]);
		double thermZ = sd * curand_normal_double(&state[i]);

		H_d[i].x += thermX;
		H_d[i].y += thermY;
		H_d[i].z += thermZ;


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

		double Hex = JEX / (MSAT * ALEN * ALEN);

		H_d[i].x += Hex * (sin(up.theta) * cos(up.phi) + sin(down.theta) * cos(down.phi) + sin(left.theta) * cos(left.phi) + sin(right.theta) * cos(right.phi) + sin(front.theta) * cos(front.phi) + sin(back.theta) * cos(back.phi));
		H_d[i].y += Hex * (sin(up.theta) * sin(up.phi) + sin(down.theta) * sin(down.phi) + sin(left.theta) * sin(left.phi) + sin(right.theta) * sin(right.phi) + sin(front.theta) * sin(front.phi) + sin(back.theta) * sin(back.phi)); 
		H_d[i].z += Hex * (cos(up.theta) + cos(down.theta) + cos(left.theta) + cos(right.theta) + cos(front.theta) + cos(back.theta));
	}
}

__global__ void mDot(double t, SphVector M[], SphVector dMdt[], int nvar, Vector H[]) {
	int i = threadIdx.x + blockDim.x * blockIdx.x;

	//Compute derivative
	if(i < nvar) {
		SphVector M_s = M[i];
		Vector H_s = H[i];

		dMdt[i].r = 0;
		dMdt[i].phi = GAMMA * ((cos(M_s.theta) * sin(M_s.phi) * H_s.y) / sin(M_s.theta) + (cos(M_s.theta) * cos(M_s.phi) * H_s.x) / sin(M_s.theta) - H_s.z) + ((ALPHA * GAMMA)/(1 + ALPHA * ALPHA)) * ((cos(M_s.phi) * H_s.y) / sin(M_s.theta) - (sin(M_s.phi) * H_s.x) / sin(M_s.theta));
		dMdt[i].theta = -GAMMA * (cos(M_s.phi) * H_s.y - sin(M_s.phi) * H_s.x) + ((ALPHA * GAMMA)/(1 + ALPHA * ALPHA)) * (cos(M_s.theta) * cos(M_s.phi) * H_s.x - H_s.z * sin(M_s.theta) + cos(M_s.theta) * sin(M_s.phi) * H_s.y);
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
	SphVector *v_d, *dv_d, *yout_d;

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

	for (int k = 0; k < nstep; k++) {

		//Copy memory to device
		//After the first timestep, the value of v and yout_d are the same. d2d memcpy is much faster than h2s, so do it instead
		if(k == 0) cudaMemcpy(v_d, v, sizeof(SphVector) * nvar, cudaMemcpyHostToDevice);
		else {
			SphVector *t_d = v_d;
			v_d = yout_d;
			yout_d = t_d;
		}

		//Launch kernel to compute H field
		computeField<<<gridDim, blockDim>>>(H_d, H, v_d, nvar, state); 

		rk4Kernel<<<ceil(nvar/VECTOR_SIZE), VECTOR_SIZE>>>(v_d, dv_d, nvar, x, h, yout_d, H_d);
		
		if(k == (nstep - 1))
			cudaMemcpy(vout, yout_d, sizeof(SphVector) * nvar, cudaMemcpyDeviceToHost);

		if ((double)(x + h) == x) 
			fprintf(stderr, "Step size too small in routine rkdumb");

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
	cudaFree(yout_d);
	cudaFree(v_d);
	cudaFree(dv_d);
	cudaFree(H_d);
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

	#if BENCHMARK
	FILE * times = fopen("times.txt", "w");
	if(times == NULL) {
		printf("error opening file: times.txt\n");
		return 1;
	}
	fprintf(times, "Time to simulate %fns\n", FIELDTIMESTEP);
	#endif

	//Initialize random number generator
	unsigned long long seed = time(NULL);
	cudaMalloc((void **)&state, sizeof(curandStateXORWOW_t) * nvar);
	initializeRandom<<<ceil(nvar/512.0), 512>>>(state, nvar, seed);
	
	//Configure shared/L1 as 48KB/16KB
	cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
	
	for(int i = 0; i < nvar; i++) {	
		vstart[i].r = MSAT;
		vstart[i].theta = 0.01;
		vstart[i].phi = 0;
	}

	Vector Happl = {0.0, 0.0, FIELDRANGE};
	endTime = FIELDTIMESTEP; 
	endTime /= 100; //Reduce host memory usage
	nstep = ((int)ceil(endTime/TIMESTEP));

	xx = (double *)malloc(sizeof(double) * (nstep + 1));
	y = (SphVector **)malloc(sizeof(SphVector *) * nvar); 
	for(int i = 0; i < nvar; i++) {
		y[i] = (SphVector *)malloc(sizeof(SphVector) * (nstep + 1));
	}

	
	bool isDecreasing = true;
	for(int i = 0; i <= (4 * (int)(FIELDRANGE/FIELDSTEP)); i++) {
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
		}
	
		#if BENCHMARK
		time_t end = time(NULL);
		fprintf(times, "%lds\n", (long)(end - start));
		fflush(times);
		#endif

		double mag = 0.0;	
		for(int k = 0; k < nvar; k++) {
			mag += (y[k][nstep].r)*cos(y[k][nstep].theta);
		}

		mag /= (double)nvar;
		fprintf(output, "%f\t%f\n", Happl.z, mag);
		fflush(output);

		//Adjust applied field strength at endTime intervals	
		if(Happl.z + FIELDRANGE < 1.0) isDecreasing = false;
		isDecreasing ? (Happl.z -= FIELDSTEP) : (Happl.z += FIELDSTEP);
	}
	//Probably don't really need these since we're about to exit the program
	free(xx);
	free(y);
	cudaFree(state);
	return 0;
}
