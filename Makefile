all: runge

runge: runge.c 
	gcc -lm -std=c99 -O3 -pedantic runge.c -o runge

dbg: runge.c
	gcc -lm -g -std=c99 runge.c -o runge

profile: runge.c
	gcc -lm -g -pg -O2 -std=c99 runge.c -o runge
	./runge
	gprof ./runge
	
clean:
	rm -f runge
	rm -f gmon.out
