//Runge-Kutta 4th Order solver
//Compile with gcc -lm runge.c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static FILE * output;

static const float ALPHA = 35186; //alpha*gamma / (1 + alpha^2)
static const float GAMMA = 1.76e7;
//static const float HX = 0;
//static const float HY = 0;
//static const float HZ = -10e3;
static const float M = 1000;
static const float TIMESTEP = .000000005;
static const float K = 1e6;

typedef struct {
	float x;
	float y;
	float z;
} Vector;

//Computes the anisotropy field and writes the result to a Vector H
float anisotropyH(const float theta, const float phi, Vector * H) {
	H->x = (1/M) * -2 * K * cosf(theta) * sinf(theta) * cosf(phi) * cosf(theta);
	H->y = (1/M) * -2 * K * cosf(theta) * sinf(theta) * sinf(phi) * cosf(theta);
	H->z = (1/M) * 2 * K * cosf(theta) * powf(sinf(theta), 2);
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

int main(int argc, char *argv[]) {
	float theta = 1/57.3; //57.3 degrees in a radian
	float phi = 0;
	float t = 0;
	float endTime;

	//The applied field, H
	Vector * applH = malloc(sizeof(Vector));
	applH->x = 0.0f;
	applH->y = 0.0f;
	applH->z = -10e3;

	output = fopen("output.txt", "w");
	if(output == NULL) {
		printf("error opening file\n");
		return 0;
	}

	if(argc < 2){
		printf("Usage: %s [t]\n", argv[0]);
		return 0;
	}
	else 
		endTime = strtof(argv[1], NULL);
	
	fprintf(output, "theta: %f\tt: %f\n", theta, t);
	
	for(t = 0; t < endTime; t += TIMESTEP) {
		theta = theta + (1.0/6.0) * (k1theta(theta, applH) + 2.0 * k2theta(theta, applH) + 2.0 * k3theta(theta, applH) + k4theta(theta, applH)) * TIMESTEP;
		//printf("theta: %f\tt: %f\n", theta, t + TIMESTEP);
		//fprintf(output, "theta: %f\tt: %f\n", theta, t + TIMESTEP);
	}
	
	fprintf(output, "phi: %f\tt: %f\n", phi, t);
	
	for(t = 0; t < endTime; t += TIMESTEP) {
		phi = phi + (1.0/6.0) * (k1phi(phi, applH) + 2.0 * k2phi(phi, applH) + 2.0 * k3phi(phi, applH) + k4phi(phi, applH)) * TIMESTEP;
		//printf("phi: %f\tt: %f\n", phi, t + TIMESTEP);
		//fprintf(output, "phi: %f\tt: %f\n", phi, t + TIMESTEP);
	}
	
	printf("M(%f) = %fx + %fy + %fz\n", endTime, M * sinf(theta) * cosf(phi), M * sinf(theta) * sinf(phi), M * cosf(theta));
	fprintf(output, "M(%f) = %fx + %fy + %fz\n", endTime, M * sinf(theta) * cosf(phi), M * sinf(theta) * sinf(phi), M * cosf(theta));
	
	fclose(output);
	free(applH);
	return 0;
}	
