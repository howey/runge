//Gaussian random variable via Marsaglia polar method
//Follows Oracle's Java implementation of nextGaussian
//http://docs.oracle.com/javase/1.4.2/docs/api/java/util/Random.html#nextGaussian

#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include <time.h>
#include <stdlib.h>

#define gaussian(mean, sd) ((mean) + (sd) * nextGaussian())

//From http://ubuntuforums.org/showthread.php?t=1717717
double nextDouble(double min, double max) {
	double range = (max - min);
	double div = RAND_MAX / range;
	return min + (rand() / div);
}

double nextGaussian() {
	static bool haveNextNextGaussian = false;
	static double nextNextGaussian;

	if (haveNextNextGaussian) {
		haveNextNextGaussian = false;
		return nextNextGaussian;
	} else {
		double v1, v2, s;
		do {
			v1 = 2 * nextDouble(-1.0, 1.0);
			v2 = 2 * nextDouble(-1.0, 1.0);
			s = v1 * v1 + v2 * v2;
		} while (s >= 1 || s == 0);
		double multiplier = sqrt(-2 * log(s) / s);
		nextNextGaussian = v2 * multiplier;
		haveNextNextGaussian = true;
		return v1 * multiplier;
	}
}

#if 0
void main() {
	srand((unsigned)time(NULL));
	//Generate a bunch of randoms
	//Use gnuplot to visually verify that they're actually gaussian distributed
	for(int i = 0; i < 100000; i++) {
		printf("%f\n", gaussian(0, 1));
		//printf("%f\n", nextGaussian());
	}
}
#endif
