//Runge-Kutta 4th Order solver
//Compile with gcc -lm runge.c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static FILE * output;

static const float ALPHA = 35186;
static const float GAMMA = 1.76e7;
static const float HX = 0;
static const float HY = 0;
static const float HZ = -10e3;
static const float M = 1000;
static const float TIMESTEP = .000000005;

float phiDot(float theta, float phi) {
	return GAMMA * ((cosf(theta) * sinf(phi) * HY) / sinf(theta) + (cosf(theta) * cosf(phi) * HX) / sinf(theta) - HZ) + ALPHA * ((cosf(phi) * HY) / sinf(theta) - (sinf(phi) * HX) / sin(theta));	
}

float thetaDot(float theta, float phi) {
	return -GAMMA * (cosf(phi) * HY - sinf(phi) * HX)\
	+ ALPHA * (cosf(theta) * cosf(phi) * HX - HZ * sinf(theta) + cosf(theta) * sinf(phi) * HY);
}

//TODO: Memoize k1, k2, k3 and k4 to increase performance
float k1theta(float theta) {
	return thetaDot(theta, 0);
}

float k2theta(float theta) {
	return thetaDot(theta + 0.5 * k1theta(theta) * TIMESTEP, 0);
}

float k3theta(float theta) {
	return thetaDot(theta + .5 * k2theta(theta) * TIMESTEP, 0);
}

float k4theta(float theta) {
	return thetaDot(theta + k3theta(theta) * TIMESTEP, 0);
}

float k1phi(float phi) {
	return phiDot(1, phi);
}

float k2phi(float phi) {
	return phiDot(1, phi+ 0.5 * k1phi(phi) * TIMESTEP);
}

float k3phi(float phi) {
	return phiDot(1, phi + .5 * k2phi(phi) * TIMESTEP);
}

float k4phi(float phi) {
	return phiDot(1, phi + k3phi(phi) * TIMESTEP);
}

int main(int argc, char *argv[]) {
	output = fopen("output.txt", "w");
	if(output == NULL) {
		printf("error opening file\n");
		return 0;
	}
	float t = 0;
	float endTime;
	if(argc < 2){
		printf("Usage: %s [t]\n", argv[0]);
		return 0;
	}
	else 
		endTime = strtof(argv[1], NULL);
	float theta = 1/57.3; //57.3 degrees in a radian
	float phi = 0;
	fprintf(output, "theta: %f\tt: %f\n", theta, t);
	for(t = 0; t < endTime; t += TIMESTEP) {
		theta = theta + (1.0/6.0) * (k1theta(theta) + 2.0 * k2theta(theta) + 2.0 * k3theta(theta) + k4theta(theta)) * TIMESTEP;
		printf("theta: %f\tt: %f\n", theta, t + TIMESTEP);
		fprintf(output, "theta: %f\tt: %f\n", theta, t + TIMESTEP);
	}
	fprintf(output, "phi: %f\tt: %f\n", phi, t);
	for(t = 0; t < endTime; t += TIMESTEP) {
		phi = phi + (1.0/6.0) * (k1phi(phi) + 2.0 * k2phi(phi) + 2.0 * k3phi(phi) + k4phi(phi)) * TIMESTEP;
		printf("phi: %f\tt: %f\n", phi, t + TIMESTEP);
		fprintf(output, "phi: %f\tt: %f\n", phi, t + TIMESTEP);
	}
	printf("M(%f) = %fx + %fy + %fz\n", endTime, M * sinf(theta) * cosf(phi), M * sinf(theta) * sinf(phi), M * cosf(theta));
	fprintf(output, "M(%f) = %fx + %fy + %fz\n", endTime, M * sinf(theta) * cosf(phi), M * sinf(theta) * sinf(phi), M * cosf(theta));
	fclose(output);
	return 0;
}	
