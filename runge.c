//Runge-Kutta 4th Order solver
//Compile with gcc -lm -std=c99 runge.c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include "marsaglia-random/mars.h"

static FILE * output;

static const double ALPHA = 35186; //alpha*gamma / (1 + alpha^2)
static const double GAMMA = 1.76e7;
//static const double M = 1000;
static const double TIMESTEP = .00000000005;
static const double K = 1e6;

typedef struct {
	double x;
	double y;
	double z;
} Vector;

typedef struct {
	double r;
	double theta;
	double phi;
} SphVector;

//Computes the anisotropy field and writes the result to a Vector H
void anisotropyH(Vector * H, const SphVector * M) {
	H->x = (1/M->r) * -2 * K * cos(M->theta) * sin(M->theta) * cos(M->phi) * cos(M->theta);
	H->y = (1/M->r) * -2 * K * cos(M->theta) * sin(M->theta) * sin(M->phi) * cos(M->theta);
	H->z = (1/M->r) * 2 * K * cos(M->theta) * pow(sin(M->theta), 2);
}

//Adds rectangular vectors A and B and stores the results in A
void addVector(Vector * A, const Vector * B) {
	(A->x) += (B->x);
	(A->y) += (B->y);
	(A->z) += (B->z);
}

double phiDot(const double theta, const double phi, const Vector * H) {
	return GAMMA * ((cos(theta) * sin(phi) * H->y) / sin(theta) + (cos(theta) * cos(phi) * H->x) / sin(theta) - H->z) + ALPHA * ((cos(phi) * H->y) / sin(theta) - (sin(phi) * H->x) / sin(theta));	
}

double thetaDot(const double theta, const double phi, const Vector * H) {
	return -GAMMA * (cos(phi) * H->y - sin(phi) * H->x)\
	+ ALPHA * (cos(theta) * cos(phi) * H->x - H->z * sin(theta) + cos(theta) * sin(phi) * H->y);
}

// Terms for RK4
double k1theta(const double theta, const Vector * H) {
	return thetaDot(theta, 0, H);
}

double k2theta(const double theta, const Vector * H) {
	return thetaDot(theta + 0.5 * k1theta(theta, H) * TIMESTEP, 0, H);
}

double k3theta(const double theta, const Vector * H) {
	return thetaDot(theta + .5 * k2theta(theta, H) * TIMESTEP, 0, H);
}

double k4theta(const double theta, const Vector * H) {
	return thetaDot(theta + k3theta(theta, H) * TIMESTEP, 0, H);
}

double k1phi(const double phi, const Vector * H) {
	return phiDot(1, phi, H);
}

double k2phi(const double phi, const Vector * H) {
	return phiDot(1, phi+ 0.5 * k1phi(phi, H) * TIMESTEP, H);
}

double k3phi(const double phi, const Vector * H) {
	return phiDot(1, phi + .5 * k2phi(phi, H) * TIMESTEP, H);
}

double k4phi(const double phi, const Vector * H) {
	return phiDot(1, phi + k3phi(phi, H) * TIMESTEP, H);
}

void simulate(const Vector * H, SphVector * M, const double endTime) {
	for(double t = 0; t < endTime; t += TIMESTEP) {
		M->theta = M->theta + (1.0/6.0) * (k1theta(M->theta, H) + 2.0 * k2theta(M->theta, H) + 2.0 * k3theta(M->theta, H) + k4theta(M->theta, H)) * TIMESTEP;
	}
	
	for(double t = 0; t < endTime; t += TIMESTEP) {
		M->phi = M->phi + (1.0/6.0) * (k1phi(M->phi, H) + 2.0 * k2phi(M->phi, H) + 2.0 * k3phi(M->phi, H) + k4phi(M->phi, H)) * TIMESTEP;
	}
}

int main(int argc, char *argv[]) {
	double endTime;
	double time;
	double sd;

	//The applied field, H
	Vector * applH = malloc(sizeof(Vector));
	applH->x = 0.0;
	applH->y = 0.0;
	applH->z = 2500.0;

	//The magnetization, M
	SphVector * M = malloc(sizeof(SphVector));
	M->r = 1000;
	M->theta = (1/57.3f); //Initial angle of 1 degree, 57.3 degrees in a radian
	M->phi = 0;
	
	//The anisotropy field
	Vector * anisH = malloc(sizeof(Vector));
	anisH->x = 0.0;
	anisH->y = 0.0;
	anisH->z = 0.0;

	//The effective field
	Vector * effH = malloc(sizeof(Vector));
	effH->x = 0.0;
	effH->y = 0.0;
	effH->z = 0.0;

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
	
	sd = (3.45e-4)/sqrt(endTime);
	bool isDecreasing = true;
	do {
		//anisotropyH(anisH, M);	

		effH->x = 0.0;
		effH->y = 0.0;
		effH->z = 0.0;

		effH->x += gaussian(0, sd);
		effH->y += gaussian(0, sd);
		effH->z += gaussian(0, sd);

		//addVector(effH, anisH);
		addVector(effH, applH);
		
		simulate(effH, M, endTime);
		if(applH->z + 2500.0 < 1.0)
			isDecreasing = false;
		isDecreasing ? (applH->z -= 50.0) : (applH->z += 50.0);
		//fprintf(output, "M(%e) = %fx + %fy + %fz\n", time, M->r * sin(M->theta) * cos(M->phi), M->r * sin(M->theta) * sin(M->phi), M->r * cos(M->theta));
		fprintf(output, "%e\t%f\n", applH->z, M->r * cos(M->theta));
		time += endTime;
	} while(applH->z - 2500.0 < 1.0);

	printf("M(%e) = %fx + %fy + %fz\n", endTime, M->r * sin(M->theta) * cos(M->phi), M->r * sin(M->theta) * sin(M->phi), M->r * cos(M->theta));
	
	fclose(output);
	
	free(applH);
	free(effH);
	free(anisH);
	free(M);
	return 0;
}	
