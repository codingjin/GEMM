all: mm4

SRC = mm4_main.cu mm4_gpu.cu

mm4: $(SRC)
	nvcc -O3 -o $@ $(SRC)

clean:
	rm -f mm4
