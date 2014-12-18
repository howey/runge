CC = gcc
FLAGS = -Werror -Wall -O2 -W -Wmissing-prototypes -Wstrict-prototypes -Wconversion -Wshadow -Wpointer-arith -Wcast-qual -Wcast-align -Wwrite-strings -fno-common -Wnested-externs -g -std=c99
LIBS = -lm

cpu: runge.o mars.o 
	$(CC) $(FLAGS) $(LIBS) runge.o mars.o -o runge

runge.o: runge.c runge.h
	$(CC) $(FLAGS) $(LIBS) runge.c -c

mars.o: mars.c runge.h
	$(CC) $(FLAGS) $(LIBS) mars.c -c

clean:
	rm -f runge runge.o mars.o
