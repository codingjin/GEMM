all: mm4

SRC = mm4_main.c mm4_par.c

mm4: $(SRC)
	gcc -O3 -fopenmp -o$@ $(SRC)

clean:
	rm -f mm4
