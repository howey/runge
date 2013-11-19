all: runge

runge: runge.c 
	gcc -lm -std=c99 -O2 -pedantic -Werror -Wall -W -Wconversion -Wshadow -Wpointer-arith -Wcast-qual -Wcast-align -Wnested-externs -fshort-enums -fno-common runge.c -o runge

dbg: runge.c
	gcc -lm -g -std=c99 runge.c -o runge

clean:
	rm -f runge
