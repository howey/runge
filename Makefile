all: runge

runge: runge.c 
	gcc -lm -std=c99 runge.c -o runge

dbg: runge.c
	gcc -lm -g -std=c99 runge.c -o runge

clean:
	rm -rf runge
