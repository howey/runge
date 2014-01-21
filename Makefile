all: runge

runge: runge.cu 
	nvcc runge.cu -o runge

#dbg: runge.c
#	gcc -lm -g -std=c99 runge.c -o runge

#profile: runge.c
#	gcc -lm -g -pg -O2 -std=c99 runge.c -o runge
#	./runge
#	gprof ./runge
	
clean:
	rm -f runge
	rm -f gmon.out
