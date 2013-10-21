//Runge-Kutta 4th Order solver
//Compile with gcc -lm -std=c99 runge.c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>

static FILE * output;

static const float ALPHA = 35186; //alpha*gamma / (1 + alpha^2)
static const float GAMMA = 1.76e7;
//static const float M = 1000;
static const float TIMESTEP = .0000000005;
static const float K = 1e6;

typedef struct {
	float x;
	float y;
	float z;
} Vector;

typedef struct {
	float r;
	float theta;
	float phi;
} SphVector;

//Computes the anisotropy field and writes the result to a Vector H
void anisotropyH(Vector * H, const SphVector * M) {
	H->x = (1/M->r) * -2 * K * cosf(M->theta) * sinf(M->theta) * cosf(M->phi) * cosf(M->theta);
	H->y = (1/M->r) * -2 * K * cosf(M->theta) * sinf(M->theta) * sinf(M->phi) * cosf(M->theta);
	H->z = (1/M->r) * 2 * K * cosf(M->theta) * powf(sinf(M->theta), 2);
}

//Adds rectangular vectors A and B and stores the results in A
void addVector(Vector * A, const Vector * B) {
	(A->x) += (B->x);
	(A->y) += (B->y);
	(A->z) += (B->z);
}

float phiDot(const float theta, const float phi, const Vector * H) {
	return GAMMA * ((cosf(theta) * sinf(phi) * H->y) / sinf(theta) + (cosf(theta) * cosf(phi) * H->x) / sinf(theta) - H->z) + ALPHA * ((cosf(phi) * H->y) / sinf(theta) - (sinf(phi) * H->x) / sin(theta));	
}

float thetaDot(const float theta, const float phi, const Vector * H) {
	return -GAMMA * (cosf(phi) * H->y - sinf(phi) * H->x)\
	+ ALPHA * (cosf(theta) * cosf(phi) * H->x - H->z * sinf(theta) + cosf(theta) * sinf(phi) * H->y);
}

// Terms for RK4
float k1theta(const float theta, const Vector * H) {
	return thetaDot(theta, 0, H);
}

float k2theta(const float theta, const Vector * H) {
	return thetaDot(theta + 0.5 * k1theta(theta, H) * TIMESTEP, 0, H);
}

float k3theta(const float theta, const Vector * H) {
	return thetaDot(theta + .5 * k2theta(theta, H) * TIMESTEP, 0, H);
}

float k4theta(const float theta, const Vector * H) {
	return thetaDot(theta + k3theta(theta, H) * TIMESTEP, 0, H);
}

float k1phi(const float phi, const Vector * H) {
	return phiDot(1, phi, H);
}

float k2phi(const float phi, const Vector * H) {
	return phiDot(1, phi+ 0.5 * k1phi(phi, H) * TIMESTEP, H);
}

float k3phi(const float phi, const Vector * H) {
	return phiDot(1, phi + .5 * k2phi(phi, H) * TIMESTEP, H);
}

float k4phi(const float phi, const Vector * H) {
	return phiDot(1, phi + k3phi(phi, H) * TIMESTEP, H);
}

void simulate(const Vector * H, SphVector * M, const float endTime) {
	for(float t = 0; t < endTime; t += TIMESTEP) {
		M->theta = M->theta + (1.0/6.0) * (k1theta(M->theta, H) + 2.0 * k2theta(M->theta, H) + 2.0 * k3theta(M->theta, H) + k4theta(M->theta, H)) * TIMESTEP;
	}
	
	for(float t = 0; t < endTime; t += TIMESTEP) {
		M->phi = M->phi + (1.0/6.0) * (k1phi(M->phi, H) + 2.0 * k2phi(M->phi, H) + 2.0 * k3phi(M->phi, H) + k4phi(M->phi, H)) * TIMESTEP;
	}
}

int main(int argc, char *argv[]) {
	//float theta = 1/57.3; //57.3 degrees in a radian
	//float phi = 0;
	float endTime;
	float time;

	//The applied field, H
	Vector * applH = malloc(sizeof(Vector));
	applH->x = 0.0f;
	applH->y = 0.0f;
	applH->z = 2500.0f;

	//The magnetization, M
	SphVector * M = malloc(sizeof(SphVector));
	M->r = 1000;
	M->theta = (1/57.3f); //Initial angle of 1 degree, 57.3 degrees in a radian
	M->phi = 0;
	
	//The anisotropy field
	Vector * anisH = malloc(sizeof(Vector));
	anisH->x = 0.0f;
	anisH->y = 0.0f;
	anisH->z = 0.0f;

	//The effective field
	Vector * effH = malloc(sizeof(Vector));
	effH->x = 0.0f;
	effH->y = 0.0f;
	effH->z = 0.0f;

	output = fopen("output.txt", "w");
	if(output == NULL) {
		printf("error opening file\n");
		return 0;
	}

	if(argc < 2){
		//Get time step in calculating hysterisis loop
		printf("Usage: %s [timestep]\n", argv[0]);
		return 0;
	}
	else 
		endTime = strtof(argv[1], NULL);
	
	bool isDecreasing = true;
	do {
		//anisotropyH(anisH, M);	

		effH->x = 0.0f;
		effH->y = 0.0f;
		effH->z = 0.0f;

		//addVector(effH, anisH);
		addVector(effH, applH);
		
		simulate(effH, M, endTime);
		if(applH->z + 2500.0f < 1.0f)
			isDecreasing = false;
		isDecreasing ? (applH->z -= 50.0f) : (applH->z += 50.0f);
		//fprintf(output, "M(%e) = %fx + %fy + %fz\n", time, M->r * sinf(M->theta) * cosf(M->phi), M->r * sinf(M->theta) * sinf(M->phi), M->r * cosf(M->theta));
		fprintf(output, "%e\t%f\n", applH->z, M->r * cosf(M->theta));
		time += endTime;
	} while(applH->z - 2500.0f < 1.0f);

	printf("M(%e) = %fx + %fy + %fz\n", endTime, M->r * sinf(M->theta) * cosf(M->phi), M->r * sinf(M->theta) * sinf(M->phi), M->r * cosf(M->theta));
	
	fclose(output);
	free(applH);
	free(M);
	return 0;
}	
