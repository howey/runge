//Runge-Kutta 4th Order solver
//Compile with gcc -lm runge.c
#include <stdio.h>
#include <math.h>

static FILE * output;

static const float ALPHA = 1;
static const float GAMMA = 1;
static const float HX = 0;
static const float HY = 0;
static const float HZ = 100;
static const float M;
static const float TIMESTEP = 1000;

float phiDot(float theta, float phi) {
	
}

float thetaDot(float theta, float phi) {
	return -GAMMA * (cosf(phi) * HY - sinf(phi) * HX)\
	+ ALPHA * (cosf(theta) * cosf(phi) * HX - HZ * sinf(theta) + cosf(theta) * sinf(phi) * HY);
}
	
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

int main() {
	#if 0
	output = fopen("output.txt", "w");
	if(output == NULL) {
		printf("error opening file\n");
		return 0;
	}
	#endif
	float t = 0;
	float theta = 1;
	for(t = 0; t < 100000000; t += TIMESTEP) {
		theta = theta + (1.0/6.0) * (k1theta(theta) + 2.0 * k2theta(theta) + 2.0 * k3theta(theta) + k4theta(theta)) * TIMESTEP;
		printf("theta: %f\tt: %f\n", theta, t);
	}
}	
