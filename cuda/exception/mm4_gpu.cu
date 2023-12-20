
#define MIN(a,b) (((a)<=(b))? (a):(b))
#define TILE_SIZE 16

__global__ void ab_gpu_i4j4db(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk)
{
	const unsigned int tx = threadIdx.x;
	const unsigned int ty = threadIdx.y;
	const unsigned int j = blockIdx.x*blockDim.x*4 + tx;
	const unsigned int i = blockIdx.y*blockDim.y*4 + ty;

	float sum0000 = 0.0;
	float sum0001 = 0.0;
	float sum0010 = 0.0;
	float sum0011 = 0.0;
	float sum0100 = 0.0;
	float sum0101 = 0.0;
	float sum0110 = 0.0;
	float sum0111 = 0.0;
	float sum1000 = 0.0;
	float sum1001 = 0.0;
	float sum1010 = 0.0;
	float sum1011 = 0.0;
	float sum1100 = 0.0;
	float sum1101 = 0.0;
	float sum1110 = 0.0;
	float sum1111 = 0.0;
	__shared__ float as00[2][TILE_SIZE][TILE_SIZE];
	__shared__ float as01[2][TILE_SIZE][TILE_SIZE];
	__shared__ float as10[2][TILE_SIZE][TILE_SIZE];
	__shared__ float as11[2][TILE_SIZE][TILE_SIZE];
	__shared__ float bs00[2][TILE_SIZE][TILE_SIZE];
	__shared__ float bs01[2][TILE_SIZE][TILE_SIZE];
	__shared__ float bs10[2][TILE_SIZE][TILE_SIZE];
	__shared__ float bs11[2][TILE_SIZE][TILE_SIZE];
	unsigned short int current = 0;

	if (i<Ni && tx<Nk)	as00[0][ty][tx] = A[i*Nk + tx];
	else	as00[0][ty][tx] = 0;

	if (i+TILE_SIZE<Ni && tx<Nk)	as01[0][ty][tx] = A[(i+TILE_SIZE)*Nk + tx];
	else	as01[0][ty][tx] = 0;

	if (i+2*TILE_SIZE<Ni && tx<Nk)	as10[0][ty][tx] = A[(i+2*TILE_SIZE)*Nk + tx];
	else	as10[0][ty][tx] = 0;

	if (i+3*TILE_SIZE<Ni && tx<Nk)	as11[0][ty][tx] = A[(i+3*TILE_SIZE)*Nk + tx];
	else	as11[0][ty][tx] = 0;

	if (j<Nj && ty<Nk)	bs00[0][ty][tx] = B[ty*Nj + j];
	else	bs00[0][ty][tx] = 0;

	if (j+TILE_SIZE<Nj && ty<Nk)	bs01[0][ty][tx] = B[ty*Nj + j+TILE_SIZE];
	else	bs01[0][ty][tx] = 0;

	if (j+2*TILE_SIZE<Nj && ty<Nk)	bs10[0][ty][tx] = B[ty*Nj + j+2*TILE_SIZE];
	else	bs10[0][ty][tx] = 0;

	if (j+3*TILE_SIZE<Nj && ty<Nk)	bs11[0][ty][tx] = B[ty*Nj + j+3*TILE_SIZE];
	else	bs11[0][ty][tx] = 0;
	__syncthreads();

	for (int kt=0; kt<Nk; kt+=TILE_SIZE) {
		if (i<Ni && j<Nj)
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum0000 += as00[current][ty][k-kt]*bs00[current][k-kt][tx];
	
		if (i<Ni && j+TILE_SIZE<Nj)
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum0001 += as00[current][ty][k-kt]*bs01[current][k-kt][tx];

		if (i<Ni && j+2*TILE_SIZE<Nj)
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum0010 += as00[current][ty][k-kt]*bs10[current][k-kt][tx];

		if (i<Ni && j+3*TILE_SIZE<Nj)
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum0011 += as00[current][ty][k-kt]*bs11[current][k-kt][tx];

		if (i+TILE_SIZE<Ni && j<Nj)
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum0100 += as01[current][ty][k-kt]*bs00[current][k-kt][tx];

		if (i+TILE_SIZE<Ni && j+TILE_SIZE<Nj)
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum0101 += as01[current][ty][k-kt]*bs01[current][k-kt][tx];

		if (i+TILE_SIZE<Ni && j+2*TILE_SIZE<Nj)
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum0110 += as01[current][ty][k-kt]*bs10[current][k-kt][tx];

		if (i+TILE_SIZE<Ni && j+3*TILE_SIZE<Nj)
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum0111 += as01[current][ty][k-kt]*bs11[current][k-kt][tx];

		if (i+2*TILE_SIZE<Ni && j<Nj)
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum1000 += as10[current][ty][k-kt]*bs00[current][k-kt][tx];

		if (i+2*TILE_SIZE<Ni && j+TILE_SIZE<Nj)
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum1001 += as10[current][ty][k-kt]*bs01[current][k-kt][tx];
		
		if (i+2*TILE_SIZE<Ni && j+2*TILE_SIZE<Nj) 
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum1010 += as10[current][ty][k-kt]*bs10[current][k-kt][tx];
		
		if (i+2*TILE_SIZE<Ni && j+3*TILE_SIZE<Nj)
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum1011 += as10[current][ty][k-kt]*bs11[current][k-kt][tx];
		
		if (i+3*TILE_SIZE<Ni && j<Nj)
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum1100 += as11[current][ty][k-kt]*bs00[current][k-kt][tx];
		
		if (i+3*TILE_SIZE<Ni && j+TILE_SIZE<Nj)
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum1101 += as11[current][ty][k-kt]*bs01[current][k-kt][tx];
		
		if (i+3*TILE_SIZE<Ni && j+2*TILE_SIZE<Nj)
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum1110 += as11[current][ty][k-kt]*bs10[current][k-kt][tx];
		
		if (i+3*TILE_SIZE<Ni && j+3*TILE_SIZE<Nj)
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum1111 += as11[current][ty][k-kt]*bs11[current][k-kt][tx];

		current = (current+1)&1;
		if (i<Ni && kt+TILE_SIZE+tx<Nk)
			as00[current][ty][tx] = A[i*Nk + kt+TILE_SIZE+tx];
		else
			as00[current][ty][tx] = 0;

		if (i+TILE_SIZE<Ni && kt+TILE_SIZE+tx<Nk)
			as01[current][ty][tx] = A[(i+TILE_SIZE)*Nk + kt+TILE_SIZE+tx];
		else
			as01[current][ty][tx] = 0;

		if (i+2*TILE_SIZE<Ni && kt+TILE_SIZE+tx<Nk)
			as10[current][ty][tx] = A[(i+2*TILE_SIZE)*Nk + kt+TILE_SIZE+tx];
		else
			as10[current][ty][tx] = 0;

		if (i+3*TILE_SIZE<Ni && kt+TILE_SIZE+tx<Nk)
			as11[current][ty][tx] = A[(i+3*TILE_SIZE)*Nk + kt+TILE_SIZE+tx];
		else
			as11[current][ty][tx] = 0;

		if (j<Nj && kt+TILE_SIZE+ty<Nk)
			bs00[current][ty][tx] = B[(kt+TILE_SIZE+ty)*Nj + j];
		else
			bs00[current][ty][tx] = 0;

		if (j+TILE_SIZE<Nj && kt+TILE_SIZE+ty<Nk)
			bs01[current][ty][tx] = B[(kt+TILE_SIZE+ty)*Nj + j+TILE_SIZE];
		else
			bs01[current][ty][tx] = 0;

		if (j+2*TILE_SIZE<Nj && kt+TILE_SIZE+ty<Nk)
			bs10[current][ty][tx] = B[(kt+TILE_SIZE+ty)*Nj + j+2*TILE_SIZE];
		else
			bs10[current][ty][tx] = 0;

		if (j+3*TILE_SIZE<Nj && kt+TILE_SIZE+ty<Nk)
			bs11[current][ty][tx] = B[(kt+TILE_SIZE+ty)*Nj + j+3*TILE_SIZE];
		else
			bs11[current][ty][tx] = 0;

		__syncthreads();
	}
	if (i<Ni && j<Nj)	C[i*Nj + j] = sum0000;
	if (i<Ni && j+TILE_SIZE<Nj)	C[i*Nj + j+TILE_SIZE] = sum0001;
	if (i<Ni && j+2*TILE_SIZE<Nj)	C[i*Nj + j+2*TILE_SIZE] = sum0010;
	if (i<Ni && j+3*TILE_SIZE<Nj)	C[i*Nj + j+3*TILE_SIZE] = sum0011;
	if (i+TILE_SIZE<Ni && j<Nj)	C[(i+TILE_SIZE)*Nj + j] = sum0100;
	if (i+TILE_SIZE<Ni && j+TILE_SIZE<Nj)	C[(i+TILE_SIZE)*Nj + j+TILE_SIZE] = sum0101;
	if (i+TILE_SIZE<Ni && j+2*TILE_SIZE<Nj)	C[(i+TILE_SIZE)*Nj + j+2*TILE_SIZE] = sum0110;
	if (i+TILE_SIZE<Ni && j+3*TILE_SIZE<Nj)	C[(i+TILE_SIZE)*Nj + j+3*TILE_SIZE] = sum0111;
	if (i+2*TILE_SIZE<Ni && j<Nj)	C[(i+2*TILE_SIZE)*Nj + j] = sum1000;
	if (i+2*TILE_SIZE<Ni && j+TILE_SIZE<Nj)	C[(i+2*TILE_SIZE)*Nj + j+TILE_SIZE] = sum1001;
	if (i+2*TILE_SIZE<Ni && j+2*TILE_SIZE<Nj)	C[(i+2*TILE_SIZE)*Nj + j+2*TILE_SIZE] = sum1010;
	if (i+2*TILE_SIZE<Ni && j+3*TILE_SIZE<Nj)	C[(i+2*TILE_SIZE)*Nj + j+3*TILE_SIZE] = sum1011;
	if (i+3*TILE_SIZE<Ni && j<Nj)	C[(i+3*TILE_SIZE)*Nj + j] = sum1100;
	if (i+3*TILE_SIZE<Ni && j+TILE_SIZE<Nj)	C[(i+3*TILE_SIZE)*Nj + j+TILE_SIZE] = sum1101;
	if (i+3*TILE_SIZE<Ni && j+2*TILE_SIZE<Nj)	C[(i+3*TILE_SIZE)*Nj + j+2*TILE_SIZE] = sum1110;
	if (i+3*TILE_SIZE<Ni && j+3*TILE_SIZE<Nj)	C[(i+3*TILE_SIZE)*Nj + j+3*TILE_SIZE] = sum1111;
}

__global__ void ab_gpu_i4j4(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk)
{
	const unsigned int tx = threadIdx.x;
	const unsigned int ty = threadIdx.y;
	const unsigned int j = blockIdx.x*blockDim.x*4 + tx;
	const unsigned int i = blockIdx.y*blockDim.y*4 + ty;

	float sum0000 = 0.0;
	float sum0001 = 0.0;
	float sum0010 = 0.0;
	float sum0011 = 0.0;
	float sum0100 = 0.0;
	float sum0101 = 0.0;
	float sum0110 = 0.0;
	float sum0111 = 0.0;
	float sum1000 = 0.0;
	float sum1001 = 0.0;
	float sum1010 = 0.0;
	float sum1011 = 0.0;
	float sum1100 = 0.0;
	float sum1101 = 0.0;
	float sum1110 = 0.0;
	float sum1111 = 0.0;
	__shared__ float as00[TILE_SIZE][TILE_SIZE];
	__shared__ float as01[TILE_SIZE][TILE_SIZE];
	__shared__ float as10[TILE_SIZE][TILE_SIZE];
	__shared__ float as11[TILE_SIZE][TILE_SIZE];
	__shared__ float bs00[TILE_SIZE][TILE_SIZE];
	__shared__ float bs01[TILE_SIZE][TILE_SIZE];
	__shared__ float bs10[TILE_SIZE][TILE_SIZE];
	__shared__ float bs11[TILE_SIZE][TILE_SIZE];

	for (int kt=0; kt<Nk; kt+=TILE_SIZE) {
		if (i<Ni && kt+tx<Nk)
			as00[ty][tx] = A[i*Nk + kt+tx];
		else
			as00[ty][tx] = 0;

		if (i+TILE_SIZE<Ni && kt+tx<Nk)
			as01[ty][tx] = A[(i+TILE_SIZE)*Nk + kt+tx];
		else
			as01[ty][tx] = 0;

		if (i+2*TILE_SIZE<Ni && kt+tx<Nk)
			as10[ty][tx] = A[(i+2*TILE_SIZE)*Nk + kt+tx];
		else
			as10[ty][tx] = 0;

		if (i+3*TILE_SIZE<Ni && kt+tx<Nk)
			as11[ty][tx] = A[(i+3*TILE_SIZE)*Nk + kt+tx];
		else
			as11[ty][tx] = 0;

		if (j<Nj && kt+ty<Nk)
			bs00[ty][tx] = B[(kt+ty)*Nj + j];
		else
			bs00[ty][tx] = 0;

		if (j+TILE_SIZE<Nj && kt+ty<Nk)
			bs01[ty][tx] = B[(kt+ty)*Nj + j+TILE_SIZE];
		else
			bs01[ty][tx] = 0;

		if (j+2*TILE_SIZE<Nj && kt+ty<Nk)
			bs10[ty][tx] = B[(kt+ty)*Nj + j+2*TILE_SIZE];
		else
			bs10[ty][tx] = 0;

		if (j+3*TILE_SIZE<Nj && kt+ty<Nk)
			bs11[ty][tx] = B[(kt+ty)*Nj + j+3*TILE_SIZE];
		else
			bs11[ty][tx] = 0;

		__syncthreads();

		if (i<Ni && j<Nj)
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum0000 += as00[ty][k-kt]*bs00[k-kt][tx];

		if (i<Ni && j+TILE_SIZE<Nj) 
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum0001 += as00[ty][k-kt]*bs01[k-kt][tx];
		
		if (i<Ni && j+2*TILE_SIZE<Nj)
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum0010 += as00[ty][k-kt]*bs10[k-kt][tx];
		
		if (i<Ni && j+3*TILE_SIZE<Nj)
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum0011 += as00[ty][k-kt]*bs11[k-kt][tx];
		
		if (i+TILE_SIZE<Ni && j<Nj)
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum0100 += as01[ty][k-kt]*bs00[k-kt][tx];
		
		if (i+TILE_SIZE<Ni && j+TILE_SIZE<Nj)
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum0101 += as01[ty][k-kt]*bs01[k-kt][tx];
		
		if (i+TILE_SIZE<Ni && j+2*TILE_SIZE<Nj)
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum0110 += as01[ty][k-kt]*bs10[k-kt][tx];
		
		if (i+TILE_SIZE<Ni && j+3*TILE_SIZE<Nj)
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum0111 += as01[ty][k-kt]*bs11[k-kt][tx];
		
		if (i+2*TILE_SIZE<Ni && j<Nj)
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum1000 += as10[ty][k-kt]*bs00[k-kt][tx];
		
		if (i+2*TILE_SIZE<Ni && j+TILE_SIZE<Nj)
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum1001 += as10[ty][k-kt]*bs01[k-kt][tx];
		
		if (i+2*TILE_SIZE<Ni && j+2*TILE_SIZE<Nj)
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum1010 += as10[ty][k-kt]*bs10[k-kt][tx];
		
		if (i+2*TILE_SIZE<Ni && j+3*TILE_SIZE<Nj)
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum1011 += as10[ty][k-kt]*bs11[k-kt][tx];
		
		if (i+3*TILE_SIZE<Ni && j<Nj)
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum1100 += as11[ty][k-kt]*bs00[k-kt][tx];
		
		if (i+3*TILE_SIZE<Ni && j+TILE_SIZE<Nj)
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum1101 += as11[ty][k-kt]*bs01[k-kt][tx];
		
		if (i+3*TILE_SIZE<Ni && j+2*TILE_SIZE<Nj)
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum1110 += as11[ty][k-kt]*bs10[k-kt][tx];
		
		if (i+3*TILE_SIZE<Ni && j+3*TILE_SIZE<Nj)
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum1111 += as11[ty][k-kt]*bs11[k-kt][tx];

		__syncthreads();
	}

	if (i<Ni && j<Nj)	C[i*Nj + j] = sum0000;
	if (i<Ni && j+TILE_SIZE<Nj)	C[i*Nj + j+TILE_SIZE] = sum0001;
	if (i<Ni && j+2*TILE_SIZE<Nj)	C[i*Nj + j+2*TILE_SIZE] = sum0010;
	if (i<Ni && j+3*TILE_SIZE<Nj)	C[i*Nj + j+3*TILE_SIZE] = sum0011;
	if (i+TILE_SIZE<Ni && j<Nj)	C[(i+TILE_SIZE)*Nj + j] = sum0100;
	if (i+TILE_SIZE<Ni && j+TILE_SIZE<Nj)	C[(i+TILE_SIZE)*Nj + j+TILE_SIZE] = sum0101;
	if (i+TILE_SIZE<Ni && j+2*TILE_SIZE<Nj)	C[(i+TILE_SIZE)*Nj + j+2*TILE_SIZE] = sum0110;
	if (i+TILE_SIZE<Ni && j+3*TILE_SIZE<Nj)	C[(i+TILE_SIZE)*Nj + j+3*TILE_SIZE] = sum0111;
	if (i+2*TILE_SIZE<Ni && j<Nj)	C[(i+2*TILE_SIZE)*Nj + j] = sum1000;
	if (i+2*TILE_SIZE<Ni && j+TILE_SIZE<Nj)	C[(i+2*TILE_SIZE)*Nj + j+TILE_SIZE] = sum1001;
	if (i+2*TILE_SIZE<Ni && j+2*TILE_SIZE<Nj)	C[(i+2*TILE_SIZE)*Nj + j+2*TILE_SIZE] = sum1010;
	if (i+2*TILE_SIZE<Ni && j+3*TILE_SIZE<Nj)	C[(i+2*TILE_SIZE)*Nj + j+3*TILE_SIZE] = sum1011;
	if (i+3*TILE_SIZE<Ni && j<Nj)	C[(i+3*TILE_SIZE)*Nj + j] = sum1100;
	if (i+3*TILE_SIZE<Ni && j+TILE_SIZE<Nj)	C[(i+3*TILE_SIZE)*Nj + j+TILE_SIZE] = sum1101;
	if (i+3*TILE_SIZE<Ni && j+2*TILE_SIZE<Nj)	C[(i+3*TILE_SIZE)*Nj + j+2*TILE_SIZE] = sum1110;
	if (i+3*TILE_SIZE<Ni && j+3*TILE_SIZE<Nj)	C[(i+3*TILE_SIZE)*Nj + j+3*TILE_SIZE] = sum1111;
}

__global__ void ab_gpu_i2j2db(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk)
{
	const unsigned int tx = threadIdx.x;
	const unsigned int ty = threadIdx.y;
	const unsigned int j = blockIdx.x*blockDim.x*2 + tx;
	const unsigned int i = blockIdx.y*blockDim.y*2 + ty;

	float sum00 = 0.0;
	float sum01 = 0.0;
	float sum10 = 0.0;
	float sum11 = 0.0;
	__shared__ float as0[2][TILE_SIZE][TILE_SIZE];
	__shared__ float as1[2][TILE_SIZE][TILE_SIZE];
	__shared__ float bs0[2][TILE_SIZE][TILE_SIZE];
	__shared__ float bs1[2][TILE_SIZE][TILE_SIZE];
	unsigned int current = 0;

	if (i<Ni && tx<Nk)
		as0[0][ty][tx] = A[i*Nk + tx];
	else
		as0[0][ty][tx] = 0;
	if (i+TILE_SIZE<Ni && tx<Nk)
		as1[0][ty][tx] = A[(i+TILE_SIZE)*Nk + tx];
	else
		as1[0][ty][tx] = 0;
	if (j<Nj && ty<Nk)
		bs0[0][ty][tx] = B[ty*Nj + j];
	else
		bs0[0][ty][tx] = 0;
	if (j+TILE_SIZE<Nj && ty<Nk)
		bs1[0][ty][tx] = B[ty*Nj + j+TILE_SIZE];
	else
		bs1[0][ty][tx] = 0;
	__syncthreads();

	for (int kt=0; kt<Nk; kt+=TILE_SIZE) {
		const unsigned int previous = current;
		current = (current+1)&1;
		if (i<Ni && j<Nj) {
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum00 += as0[previous][ty][k-kt]*bs0[previous][k-kt][tx];
		}

		if (i<Ni && j+TILE_SIZE<Nj) {
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum01 += as0[previous][ty][k-kt]*bs1[previous][k-kt][tx];
		}

		if (i+TILE_SIZE<Ni && j<Nj) {
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum10 += as1[previous][ty][k-kt]*bs0[previous][k-kt][tx];
		}

		if (i+TILE_SIZE<Ni && j+TILE_SIZE<Nj) {
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum11 += as1[previous][ty][k-kt]*bs1[previous][k-kt][tx];
		}
		if (i<Ni && kt+TILE_SIZE+tx<Nk)
			as0[current][ty][tx] = A[i*Nk + kt+TILE_SIZE+tx];
		else
			as0[current][ty][tx] = 0;
		if (i+TILE_SIZE<Ni && kt+TILE_SIZE+tx<Nk)
			as1[current][ty][tx] = A[(i+TILE_SIZE)*Nk + kt+TILE_SIZE+tx];
		else
			as1[current][ty][tx] = 0;
		if (j<Nj && kt+TILE_SIZE+ty<Nk)
			bs0[current][ty][tx] = B[(kt+TILE_SIZE+ty)*Nj + j];
		else
			bs0[current][ty][tx] = 0;
		if (j+TILE_SIZE<Nj && kt+TILE_SIZE+ty<Nk)
			bs1[current][ty][tx] = B[(kt+TILE_SIZE+ty)*Nj + j+TILE_SIZE];
		else
			bs1[current][ty][tx] = 0;
		__syncthreads();
	}

	if (i<Ni && j<Nj)	C[i*Nj + j] = sum00;
	if (i<Ni && j+TILE_SIZE<Nj)	C[i*Nj + j+TILE_SIZE] = sum01;
	if (i+TILE_SIZE<Ni && j<Nj)	C[(i+TILE_SIZE)*Nj + j] = sum10;
	if (i+TILE_SIZE<Ni && j+TILE_SIZE<Nj)	C[(i+TILE_SIZE)*Nj + j+TILE_SIZE] = sum11;
}

__global__ void ab_gpu_db(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk)
{
	const unsigned int tx = threadIdx.x;
	const unsigned int ty = threadIdx.y;
	const unsigned int j = blockIdx.x*blockDim.x + tx;
	const unsigned int i = blockIdx.y*blockDim.y + ty;
	float sum = 0.0;
	unsigned int current = 0;

	__shared__ float as[2][TILE_SIZE][TILE_SIZE];
	__shared__ float bs[2][TILE_SIZE][TILE_SIZE];
	if (i<Ni && tx<Nk)	as[0][ty][tx] = A[i*Nk + tx];
	else				as[0][ty][tx] = 0;
	if (j<Nj && ty<Nk)	bs[0][ty][tx] = B[ty*Nj + j];
	else				bs[0][ty][tx] = 0;
	__syncthreads();

	for (int kt=0; kt<Nk; kt+=TILE_SIZE) {
		if (i<Ni && j<Nj)
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum += as[current][ty][k-kt]*bs[current][k-kt][tx];

		current = (current+1)&1;
		if (i<Ni && kt+TILE_SIZE+tx<Nk)
			as[current][ty][tx] = A[i*Nk + kt+TILE_SIZE+tx];
		else
			as[current][ty][tx] = 0;
		if (j<Nj && kt+TILE_SIZE+ty<Nk)
			bs[current][ty][tx] = B[(kt+TILE_SIZE+ty)*Nj + j];
		else
			bs[current][ty][tx] = 0;
		__syncthreads();
	}
	if (i<Ni && j<Nj)	C[i*Nj + j] = sum;
}

__global__ void ab_gpu(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk)
{
	const unsigned int tx = threadIdx.x;
	const unsigned int ty = threadIdx.y;
	const unsigned int j = blockIdx.x*blockDim.x + tx;
	const unsigned int i = blockIdx.y*blockDim.y + ty;
	float sum = 0.0;
	unsigned int current = 0;

	__shared__ float as[2][TILE_SIZE][TILE_SIZE];
	__shared__ float bs[2][TILE_SIZE][TILE_SIZE];
	if (i<Ni && tx<Nk)	as[0][ty][tx] = A[i*Nk + tx];
	else	as[0][ty][tx] = 0;
	if (j<Nj && ty<Nk)	bs[0][ty][tx] = B[ty*Nj + j];
	else	bs[0][ty][tx] = 0;
	__syncthreads();

	for (int kt=0; kt<Nk; kt+=TILE_SIZE) {
		if (i<Ni && j<Nj) {
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum += as[current][ty][k-kt]*bs[current][k-kt][tx];
		}
		current = (current+1)&1;
		if (i<Ni && kt+TILE_SIZE+tx<Nk)
			as[current][ty][tx] = A[i*Nk + kt+TILE_SIZE+tx];
		else
			as[current][ty][tx] = 0;
		if (j<Nj && kt+TILE_SIZE+ty<Nk)
			bs[current][ty][tx] = B[(kt+TILE_SIZE+ty)*Nj + j];
		else
			bs[current][ty][tx] = 0;
		__syncthreads();
	}
	if (i<Ni && j<Nj)	C[i*Nj + j] = sum;
}

__global__ void aTb_gpu(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk)
{
	const unsigned int tx = threadIdx.x;
	const unsigned int ty = threadIdx.y;
	const unsigned int i = blockIdx.y*blockDim.y*4 + ty;
	const unsigned int j = blockIdx.x*blockDim.x*4 + tx;

	__shared__ float as00[TILE_SIZE][TILE_SIZE];
	__shared__ float as01[TILE_SIZE][TILE_SIZE];
	__shared__ float as10[TILE_SIZE][TILE_SIZE];
	__shared__ float as11[TILE_SIZE][TILE_SIZE];
	__shared__ float bs00[TILE_SIZE][TILE_SIZE];
	__shared__ float bs01[TILE_SIZE][TILE_SIZE];
	__shared__ float bs10[TILE_SIZE][TILE_SIZE];
	__shared__ float bs11[TILE_SIZE][TILE_SIZE];
	float sum0000 = 0.0;
	float sum0001 = 0.0;
	float sum0010 = 0.0;
	float sum0011 = 0.0;
	float sum0100 = 0.0;
	float sum0101 = 0.0;
	float sum0110 = 0.0;
	float sum0111 = 0.0;
	float sum1000 = 0.0;
	float sum1001 = 0.0;
	float sum1010 = 0.0;
	float sum1011 = 0.0;
	float sum1100 = 0.0;
	float sum1101 = 0.0;
	float sum1110 = 0.0;
	float sum1111 = 0.0;

	for (int kt=0; kt<Nk; kt+=TILE_SIZE) {
		if (i<Ni && (kt+tx)<Nk)
			as00[tx][ty] = A[(kt+tx)*Ni + i];
		else
			as00[tx][ty] = 0.0;
		if (i+TILE_SIZE<Ni && (kt+tx)<Nk)
			as01[tx][ty] = A[(kt+tx)*Ni + i+TILE_SIZE];
		else
			as01[tx][ty] = 0.0;
		if (i+2*TILE_SIZE<Ni && (kt+tx)<Nk)
			as10[tx][ty] = A[(kt+tx)*Ni + i+2*TILE_SIZE];
		else
			as10[tx][ty] = 0.0;
		if (i+3*TILE_SIZE<Ni && (kt+tx)<Nk)
			as11[tx][ty] = A[(kt+tx)*Ni + i+3*TILE_SIZE];
		else
			as11[tx][ty] = 0.0;

		if (j<Nj && (kt+ty)<Nk)
			bs00[ty][tx] = B[(kt+ty)*Nj + j];
		else
			bs00[ty][tx] = 0.0;
		if (j+TILE_SIZE<Nj && (kt+ty)<Nk)
			bs01[ty][tx] = B[(kt+ty)*Nj + j+TILE_SIZE];
		else
			bs01[ty][tx] = 0.0;
		if (j+2*TILE_SIZE<Nj && (kt+ty)<Nk)
			bs10[ty][tx] = B[(kt+ty)*Nj + j+2*TILE_SIZE];
		else
			bs10[ty][tx] = 0.0;
		if (j+3*TILE_SIZE<Nj && (kt+ty)<Nk)
			bs11[ty][tx] = B[(kt+ty)*Nj + j+3*TILE_SIZE];
		else
			bs11[ty][tx] = 0.0;
		__syncthreads();

		if (i<Ni && j<Nj) {
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum0000 += as00[k-kt][ty]*bs00[k-kt][tx];
		}
		if (i<Ni && j+TILE_SIZE<Nj) {
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum0001 += as00[k-kt][ty]*bs01[k-kt][tx];
		}
		if (i<Ni && j+2*TILE_SIZE<Nj) {
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum0010 += as00[k-kt][ty]*bs10[k-kt][tx];
		}
		if (i<Ni && j+3*TILE_SIZE<Nj) {
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum0011 += as00[k-kt][ty]*bs11[k-kt][tx];
		}
		if (i+TILE_SIZE<Ni && j<Nj) {
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum0100 += as01[k-kt][ty]*bs00[k-kt][tx];
		}
		if (i+TILE_SIZE<Ni && j+TILE_SIZE<Nj) {
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum0101 += as01[k-kt][ty]*bs01[k-kt][tx];
		}
		if (i+TILE_SIZE<Ni && j+2*TILE_SIZE<Nj) {
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum0110 += as01[k-kt][ty]*bs10[k-kt][tx];
		}
		if (i+TILE_SIZE<Ni && j+3*TILE_SIZE<Nj) {
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum0111 += as01[k-kt][ty]*bs11[k-kt][tx];
		}
		if (i+2*TILE_SIZE<Ni && j<Nj) {
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum1000 += as10[k-kt][ty]*bs00[k-kt][tx];
		}
		if (i+2*TILE_SIZE<Ni && j+TILE_SIZE<Nj) {
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum1001 += as10[k-kt][ty]*bs01[k-kt][tx];
		}
		if (i+2*TILE_SIZE<Ni && j+2*TILE_SIZE<Nj) {
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum1010 += as10[k-kt][ty]*bs10[k-kt][tx];
		}
		if (i+2*TILE_SIZE<Ni && j+3*TILE_SIZE<Nj) {
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum1011 += as10[k-kt][ty]*bs11[k-kt][tx];
		}
		if (i+3*TILE_SIZE<Ni && j<Nj) {
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum1100 += as11[k-kt][ty]*bs00[k-kt][tx];
		}
		if (i+3*TILE_SIZE<Ni && j+TILE_SIZE<Nj) {
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum1101 += as11[k-kt][ty]*bs01[k-kt][tx];
		}
		if (i+3*TILE_SIZE<Ni && j+2*TILE_SIZE<Nj) {
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum1110 += as11[k-kt][ty]*bs10[k-kt][tx];
		}
		if (i+3*TILE_SIZE<Ni && j+3*TILE_SIZE<Nj) {
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum1111 += as11[k-kt][ty]*bs11[k-kt][tx];
		}
		__syncthreads();
	}

	if (i<Ni && j<Nj)	C[i*Nj + j] = sum0000;
	if (i<Ni && j+TILE_SIZE<Nj)	C[i*Nj + j+TILE_SIZE] = sum0001;
	if (i<Ni && j+2*TILE_SIZE<Nj)	C[i*Nj + j+2*TILE_SIZE] = sum0010;
	if (i<Ni && j+3*TILE_SIZE<Nj)	C[i*Nj + j+3*TILE_SIZE] = sum0011;
	if (i+TILE_SIZE<Ni && j<Nj)	C[(i+TILE_SIZE)*Nj + j] = sum0100;
	if (i+TILE_SIZE<Ni && j+TILE_SIZE<Nj)	C[(i+TILE_SIZE)*Nj + j+TILE_SIZE] = sum0101;
	if (i+TILE_SIZE<Ni && j+2*TILE_SIZE<Nj)	C[(i+TILE_SIZE)*Nj + j+2*TILE_SIZE] = sum0110;
	if (i+TILE_SIZE<Ni && j+3*TILE_SIZE<Nj)	C[(i+TILE_SIZE)*Nj + j+3*TILE_SIZE] = sum0111;
	if (i+2*TILE_SIZE<Ni && j<Nj)	C[(i+2*TILE_SIZE)*Nj + j] = sum1000;
	if (i+2*TILE_SIZE<Ni && j+TILE_SIZE<Nj)	C[(i+2*TILE_SIZE)*Nj + j+TILE_SIZE] = sum1001;
	if (i+2*TILE_SIZE<Ni && j+2*TILE_SIZE<Nj)	C[(i+2*TILE_SIZE)*Nj + j+2*TILE_SIZE] = sum1010;
	if (i+2*TILE_SIZE<Ni && j+3*TILE_SIZE<Nj)	C[(i+2*TILE_SIZE)*Nj + j+3*TILE_SIZE] = sum1011;
	if (i+3*TILE_SIZE<Ni && j<Nj)	C[(i+3*TILE_SIZE)*Nj + j] = sum1100;
	if (i+3*TILE_SIZE<Ni && j+TILE_SIZE<Nj)	C[(i+3*TILE_SIZE)*Nj + j+TILE_SIZE] = sum1101;
	if (i+3*TILE_SIZE<Ni && j+2*TILE_SIZE<Nj)	C[(i+3*TILE_SIZE)*Nj + j+2*TILE_SIZE] = sum1110;
	if (i+3*TILE_SIZE<Ni && j+3*TILE_SIZE<Nj)	C[(i+3*TILE_SIZE)*Nj + j+3*TILE_SIZE] = sum1111;
}

__global__ void aTb_gpu_i4j4(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk)
{
	const unsigned int tx = threadIdx.x;
	const unsigned int ty = threadIdx.y;
	const unsigned int i = blockIdx.y*blockDim.y*4 + ty;
	const unsigned int j = blockIdx.x*blockDim.x*4 + tx;

	__shared__ float as00[TILE_SIZE][TILE_SIZE];
	__shared__ float as01[TILE_SIZE][TILE_SIZE];
	__shared__ float as10[TILE_SIZE][TILE_SIZE];
	__shared__ float as11[TILE_SIZE][TILE_SIZE];
	__shared__ float bs00[TILE_SIZE][TILE_SIZE];
	__shared__ float bs01[TILE_SIZE][TILE_SIZE];
	__shared__ float bs10[TILE_SIZE][TILE_SIZE];
	__shared__ float bs11[TILE_SIZE][TILE_SIZE];
	float sum0000 = 0.0;
	float sum0001 = 0.0;
	float sum0010 = 0.0;
	float sum0011 = 0.0;
	float sum0100 = 0.0;
	float sum0101 = 0.0;
	float sum0110 = 0.0;
	float sum0111 = 0.0;
	float sum1000 = 0.0;
	float sum1001 = 0.0;
	float sum1010 = 0.0;
	float sum1011 = 0.0;
	float sum1100 = 0.0;
	float sum1101 = 0.0;
	float sum1110 = 0.0;
	float sum1111 = 0.0;

	for (int kt=0; kt<Nk; kt+=TILE_SIZE) {
		if (i<Ni && (kt+tx)<Nk)	as00[tx][ty] = A[(kt+tx)*Ni + i];
		else	as00[tx][ty] = 0.0;
		if (i+TILE_SIZE<Ni && (kt+tx)<Nk)	as01[tx][ty] = A[(kt+tx)*Ni + i+TILE_SIZE];
		else	as01[tx][ty] = 0.0;
		if (i+2*TILE_SIZE<Ni && (kt+tx)<Nk)	as10[tx][ty] = A[(kt+tx)*Ni + i+2*TILE_SIZE];
		else	as10[tx][ty] = 0.0;
		if (i+3*TILE_SIZE<Ni && (kt+tx)<Nk)	as11[tx][ty] = A[(kt+tx)*Ni + i+3*TILE_SIZE];
		else	as11[tx][ty] = 0.0;

		if (j<Nj && (kt+ty)<Nk)	bs00[ty][tx] = B[(kt+ty)*Nj + j];
		else	bs00[ty][tx] = 0.0;
		if (j+TILE_SIZE<Nj && (kt+ty)<Nk)	bs01[ty][tx] = B[(kt+ty)*Nj + j+TILE_SIZE];
		else	bs01[ty][tx] = 0.0;
		if (j+2*TILE_SIZE<Nj && (kt+ty)<Nk)	bs10[ty][tx] = B[(kt+ty)*Nj + j+2*TILE_SIZE];
		else	bs10[ty][tx] = 0.0;
		if (j+3*TILE_SIZE<Nj && (kt+ty)<Nk)	bs11[ty][tx] = B[(kt+ty)*Nj + j+3*TILE_SIZE];
		else	bs11[ty][tx] = 0.0;
		__syncthreads();

		if (i<Ni && j<Nj) {
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum0000 += as00[k-kt][ty]*bs00[k-kt][tx];
		}
		if (i<Ni && j+TILE_SIZE<Nj) {
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum0001 += as00[k-kt][ty]*bs01[k-kt][tx];
		}
		if (i<Ni && j+2*TILE_SIZE<Nj) {
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum0010 += as00[k-kt][ty]*bs10[k-kt][tx];
		}
		if (i<Ni && j+3*TILE_SIZE<Nj) {
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum0011 += as00[k-kt][ty]*bs11[k-kt][tx];
		}

		if (i+TILE_SIZE<Ni && j<Nj) {
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum0100 += as01[k-kt][ty]*bs00[k-kt][tx];
		}
		if (i+TILE_SIZE<Ni && j+TILE_SIZE<Nj) {
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum0101 += as01[k-kt][ty]*bs01[k-kt][tx];
		}
		if (i+TILE_SIZE<Ni && j+2*TILE_SIZE<Nj) {
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum0110 += as01[k-kt][ty]*bs10[k-kt][tx];
		}
		if (i+TILE_SIZE<Ni && j+3*TILE_SIZE<Nj) {
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum0111 += as01[k-kt][ty]*bs11[k-kt][tx];
		}
		if (i+2*TILE_SIZE<Ni && j<Nj) {
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum1000 += as10[k-kt][ty]*bs00[k-kt][tx];
		}
		if (i+2*TILE_SIZE<Ni && j+TILE_SIZE<Nj) {
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum1001 += as10[k-kt][ty]*bs01[k-kt][tx];
		}
		if (i+2*TILE_SIZE<Ni && j+2*TILE_SIZE<Nj) {
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum1010 += as10[k-kt][ty]*bs10[k-kt][tx];
		}
		if (i+2*TILE_SIZE<Ni && j+3*TILE_SIZE<Nj) {
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum1011 += as10[k-kt][ty]*bs11[k-kt][tx];
		}

		if (i+3*TILE_SIZE<Ni && j<Nj) {
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum1100 += as11[k-kt][ty]*bs00[k-kt][tx];
		}
		if (i+3*TILE_SIZE<Ni && j+TILE_SIZE<Nj) {
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum1101 += as11[k-kt][ty]*bs01[k-kt][tx];
		}
		if (i+3*TILE_SIZE<Ni && j+2*TILE_SIZE<Nj) {
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum1110 += as11[k-kt][ty]*bs10[k-kt][tx];
		}
		if (i+3*TILE_SIZE<Ni && j+3*TILE_SIZE<Nj) {
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum1111 += as11[k-kt][ty]*bs11[k-kt][tx];
		}
		__syncthreads();
	}
	if (i<Ni && j<Nj)	C[i*Nj + j] = sum0000;
	if (i<Ni && j+TILE_SIZE<Nj)	C[i*Nj + j+TILE_SIZE] = sum0001;
	if (i<Ni && j+2*TILE_SIZE<Nj)	C[i*Nj + j+2*TILE_SIZE] = sum0010;
	if (i<Ni && j+3*TILE_SIZE<Nj)	C[i*Nj + j+3*TILE_SIZE] = sum0011;
	if (i+TILE_SIZE<Ni && j<Nj)	C[(i+TILE_SIZE)*Nj + j] = sum0100;
	if (i+TILE_SIZE<Ni && j+TILE_SIZE<Nj)	C[(i+TILE_SIZE)*Nj + j+TILE_SIZE] = sum0101;
	if (i+TILE_SIZE<Ni && j+2*TILE_SIZE<Nj)	C[(i+TILE_SIZE)*Nj + j+2*TILE_SIZE] = sum0110;
	if (i+TILE_SIZE<Ni && j+3*TILE_SIZE<Nj)	C[(i+TILE_SIZE)*Nj + j+3*TILE_SIZE] = sum0111;
	if (i+2*TILE_SIZE<Ni && j<Nj)	C[(i+2*TILE_SIZE)*Nj + j] = sum1000;
	if (i+2*TILE_SIZE<Ni && j+TILE_SIZE<Nj)	C[(i+2*TILE_SIZE)*Nj + j+TILE_SIZE] = sum1001;
	if (i+2*TILE_SIZE<Ni && j+2*TILE_SIZE<Nj)	C[(i+2*TILE_SIZE)*Nj + j+2*TILE_SIZE] = sum1010;
	if (i+2*TILE_SIZE<Ni && j+3*TILE_SIZE<Nj)	C[(i+2*TILE_SIZE)*Nj + j+3*TILE_SIZE] = sum1011;
	if (i+3*TILE_SIZE<Ni && j<Nj)	C[(i+3*TILE_SIZE)*Nj + j] = sum1100;
	if (i+3*TILE_SIZE<Ni && j+TILE_SIZE<Nj)	C[(i+3*TILE_SIZE)*Nj + j+TILE_SIZE] = sum1101;
	if (i+3*TILE_SIZE<Ni && j+2*TILE_SIZE<Nj)	C[(i+3*TILE_SIZE)*Nj + j+2*TILE_SIZE] = sum1110;
	if (i+3*TILE_SIZE<Ni && j+3*TILE_SIZE<Nj)	C[(i+3*TILE_SIZE)*Nj + j+3*TILE_SIZE] = sum1111;
}

__global__ void aTb_gpu_i4j4db(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk)
{
	const unsigned int tx = threadIdx.x;
	const unsigned int ty = threadIdx.y;
	const unsigned int i = blockIdx.y*blockDim.y*4 + ty;
	const unsigned int j = blockIdx.x*blockDim.x*4 + tx;

	__shared__ float as00[2][TILE_SIZE][TILE_SIZE];
	__shared__ float as01[2][TILE_SIZE][TILE_SIZE];
	__shared__ float as10[2][TILE_SIZE][TILE_SIZE];
	__shared__ float as11[2][TILE_SIZE][TILE_SIZE];
	__shared__ float bs00[2][TILE_SIZE][TILE_SIZE];
	__shared__ float bs01[2][TILE_SIZE][TILE_SIZE];
	__shared__ float bs10[2][TILE_SIZE][TILE_SIZE];
	__shared__ float bs11[2][TILE_SIZE][TILE_SIZE];
	float sum0000 = 0.0;
	float sum0001 = 0.0;
	float sum0010 = 0.0;
	float sum0011 = 0.0;
	float sum0100 = 0.0;
	float sum0101 = 0.0;
	float sum0110 = 0.0;
	float sum0111 = 0.0;
	float sum1000 = 0.0;
	float sum1001 = 0.0;
	float sum1010 = 0.0;
	float sum1011 = 0.0;
	float sum1100 = 0.0;
	float sum1101 = 0.0;
	float sum1110 = 0.0;
	float sum1111 = 0.0;
	unsigned short int current = 0;

	if (i<Ni && tx<Nk)		as00[0][tx][ty] = A[tx*Ni + i];
	else					as00[0][tx][ty] = 0.0;
	if (i+TILE_SIZE<Ni && tx<Nk)	as01[0][tx][ty] = A[tx*Ni + i+TILE_SIZE];
	else							as01[0][tx][ty] = 0.0;
	if (i+2*TILE_SIZE<Ni && tx<Nk)	as10[0][tx][ty] = A[tx*Ni + i+2*TILE_SIZE];
	else							as10[0][tx][ty] = 0.0;
	if (i+3*TILE_SIZE<Ni && tx<Nk)	as11[0][tx][ty] = A[tx*Ni + i+3*TILE_SIZE];
	else							as11[0][tx][ty] = 0.0;
	if (j<Nj && ty<Nk)		bs00[0][ty][tx] = B[ty*Nj + j];
	else					bs00[0][ty][tx] = 0.0;
	if (j+TILE_SIZE<Nj && ty<Nk)	bs01[0][ty][tx] = B[ty*Nj + j+TILE_SIZE];
	else							bs01[0][ty][tx] = 0.0;
	if (j+2*TILE_SIZE<Nj && ty<Nk)	bs10[0][ty][tx] = B[ty*Nj + j+2*TILE_SIZE];
	else							bs10[0][ty][tx] = 0.0;
	if (j+3*TILE_SIZE<Nj && ty<Nk)	bs11[0][ty][tx] = B[ty*Nj + j+3*TILE_SIZE];
	else							bs11[0][ty][tx] = 0.0;
	__syncthreads();

	for (int kt=0; kt<Nk; kt+=TILE_SIZE) {
		if (i<Ni && j<Nj) {
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum0000 += as00[current][k-kt][ty]*bs00[current][k-kt][tx];
		}
		if (i<Ni && j+TILE_SIZE<Nj) {
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum0001 += as00[current][k-kt][ty]*bs01[current][k-kt][tx];
		}
		if (i<Ni && j+2*TILE_SIZE<Nj) {
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum0010 += as00[current][k-kt][ty]*bs10[current][k-kt][tx];
		}
		if (i<Ni && j+3*TILE_SIZE<Nj) {
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum0011 += as00[current][k-kt][ty]*bs11[current][k-kt][tx];
		}
		if (i+TILE_SIZE<Ni && j<Nj) {
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum0100 += as01[current][k-kt][ty]*bs00[current][k-kt][tx];
		}
		if (i+TILE_SIZE<Ni && j+TILE_SIZE<Nj) {
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum0101 += as01[current][k-kt][ty]*bs01[current][k-kt][tx];
		}
		if (i+TILE_SIZE<Ni && j+2*TILE_SIZE<Nj) {
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum0110 += as01[current][k-kt][ty]*bs10[current][k-kt][tx];
		}
		if (i+TILE_SIZE<Ni && j+3*TILE_SIZE<Nj) {
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum0111 += as01[current][k-kt][ty]*bs11[current][k-kt][tx];
		}
		if (i+2*TILE_SIZE<Ni && j<Nj) {
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum1000 += as10[current][k-kt][ty]*bs00[current][k-kt][tx];
		}
		if (i+2*TILE_SIZE<Ni && j+TILE_SIZE<Nj) {
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum1001 += as10[current][k-kt][ty]*bs01[current][k-kt][tx];
		}
		if (i+2*TILE_SIZE<Ni && j+2*TILE_SIZE<Nj) {
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum1010 += as10[current][k-kt][ty]*bs10[current][k-kt][tx];
		}
		if (i+2*TILE_SIZE<Ni && j+3*TILE_SIZE<Nj) {
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum1011 += as10[current][k-kt][ty]*bs11[current][k-kt][tx];
		}
		if (i+3*TILE_SIZE<Ni && j<Nj) {
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum1100 += as11[current][k-kt][ty]*bs00[current][k-kt][tx];
		}
		if (i+3*TILE_SIZE<Ni && j+TILE_SIZE<Nj) {
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum1101 += as11[current][k-kt][ty]*bs01[current][k-kt][tx];
		}
		if (i+3*TILE_SIZE<Ni && j+2*TILE_SIZE<Nj) {
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum1110 += as11[current][k-kt][ty]*bs10[current][k-kt][tx];
		}
		if (i+3*TILE_SIZE<Ni && j+3*TILE_SIZE<Nj) {
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum1111 += as11[current][k-kt][ty]*bs11[current][k-kt][tx];
		}
		current = (current+1)&1;
		if (i<Ni && (kt+TILE_SIZE+tx)<Nk)		as00[current][tx][ty] = A[(kt+TILE_SIZE+tx)*Ni + i];
		else									as00[current][tx][ty] = 0.0;
		if (i+TILE_SIZE<Ni && (kt+TILE_SIZE+tx)<Nk)		as01[current][tx][ty] = A[(kt+TILE_SIZE+tx)*Ni + i+TILE_SIZE];
		else											as01[current][tx][ty] = 0.0;
		if (i+2*TILE_SIZE<Ni && (kt+TILE_SIZE+tx)<Nk)	as10[current][tx][ty] = A[(kt+TILE_SIZE+tx)*Ni + i+2*TILE_SIZE];
		else											as10[current][tx][ty] = 0.0;
		if (i+3*TILE_SIZE<Ni && (kt+TILE_SIZE+tx)<Nk)	as11[current][tx][ty] = A[(kt+TILE_SIZE+tx)*Ni + i+3*TILE_SIZE];
		else											as11[current][tx][ty] = 0.0;

		if (j<Nj && (kt+TILE_SIZE+ty)<Nk)		bs00[current][ty][tx] = B[(kt+TILE_SIZE+ty)*Nj + j];
		else									bs00[current][ty][tx] = 0.0;
		if (j+TILE_SIZE<Nj && (kt+TILE_SIZE+ty)<Nk)		bs01[current][ty][tx] = B[(kt+TILE_SIZE+ty)*Nj + j+TILE_SIZE];
		else											bs01[current][ty][tx] = 0.0;
		if (j+2*TILE_SIZE<Nj && (kt+TILE_SIZE+ty)<Nk)	bs10[current][ty][tx] = B[(kt+TILE_SIZE+ty)*Nj + j+2*TILE_SIZE];
		else											bs10[current][ty][tx] = 0.0;
		if (j+3*TILE_SIZE<Nj && (kt+TILE_SIZE+ty)<Nk)	bs11[current][ty][tx] = B[(kt+TILE_SIZE+ty)*Nj + j+3*TILE_SIZE];
		else											bs11[current][ty][tx] = 0.0;
		__syncthreads();
	}
	if (i<Ni && j<Nj)				C[i*Nj + j] = sum0000;
	if (i<Ni && j+TILE_SIZE<Nj)		C[i*Nj + j+TILE_SIZE] = sum0001;
	if (i<Ni && j+2*TILE_SIZE<Nj)	C[i*Nj + j+2*TILE_SIZE] = sum0010;
	if (i<Ni && j+3*TILE_SIZE<Nj)	C[i*Nj + j+3*TILE_SIZE] = sum0011;
	if (i+TILE_SIZE<Ni && j<Nj)				C[(i+TILE_SIZE)*Nj + j] = sum0100;
	if (i+TILE_SIZE<Ni && j+TILE_SIZE<Nj)	C[(i+TILE_SIZE)*Nj + j+TILE_SIZE] = sum0101;
	if (i+TILE_SIZE<Ni && j+2*TILE_SIZE<Nj)	C[(i+TILE_SIZE)*Nj + j+2*TILE_SIZE] = sum0110;
	if (i+TILE_SIZE<Ni && j+3*TILE_SIZE<Nj)	C[(i+TILE_SIZE)*Nj + j+3*TILE_SIZE] = sum0111;
	if (i+2*TILE_SIZE<Ni && j<Nj)				C[(i+2*TILE_SIZE)*Nj + j] = sum1000;
	if (i+2*TILE_SIZE<Ni && j+TILE_SIZE<Nj)		C[(i+2*TILE_SIZE)*Nj + j+TILE_SIZE] = sum1001;
	if (i+2*TILE_SIZE<Ni && j+2*TILE_SIZE<Nj)	C[(i+2*TILE_SIZE)*Nj + j+2*TILE_SIZE] = sum1010;
	if (i+2*TILE_SIZE<Ni && j+3*TILE_SIZE<Nj)	C[(i+2*TILE_SIZE)*Nj + j+3*TILE_SIZE] = sum1011;
	if (i+3*TILE_SIZE<Ni && j<Nj)				C[(i+3*TILE_SIZE)*Nj + j] = sum1100;
	if (i+3*TILE_SIZE<Ni && j+TILE_SIZE<Nj)		C[(i+3*TILE_SIZE)*Nj + j+TILE_SIZE] = sum1101;
	if (i+3*TILE_SIZE<Ni && j+2*TILE_SIZE<Nj)	C[(i+3*TILE_SIZE)*Nj + j+2*TILE_SIZE] = sum1110;
	if (i+3*TILE_SIZE<Ni && j+3*TILE_SIZE<Nj)	C[(i+3*TILE_SIZE)*Nj + j+3*TILE_SIZE] = sum1111;
}

__global__ void aTb_gpu_i2j2db(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk)
{
	const unsigned int tx = threadIdx.x;
	const unsigned int ty = threadIdx.y;
	const unsigned int i = blockIdx.y*blockDim.y*2 + ty;
	const unsigned int j = blockIdx.x*blockDim.x*2 + tx;

	__shared__ float as0[2][TILE_SIZE][TILE_SIZE];
	__shared__ float as1[2][TILE_SIZE][TILE_SIZE];
	__shared__ float bs0[2][TILE_SIZE][TILE_SIZE];
	__shared__ float bs1[2][TILE_SIZE][TILE_SIZE];
	float sum00 = 0.0;
	float sum01 = 0.0;
	float sum10 = 0.0;
	float sum11 = 0.0;
	unsigned short int current = 0;

	if (i<Ni && tx<Nk)	as0[0][tx][ty] = A[tx*Ni + i];
	else				as0[0][tx][ty] = 0.0;
	if (i+TILE_SIZE<Ni && tx<Nk)	as1[0][tx][ty] = A[tx*Ni + i+TILE_SIZE];
	else							as1[0][tx][ty] = 0.0;
	if (j<Nj && ty<Nk)	bs0[0][ty][tx] = B[ty*Nj + j];
	else				bs0[0][ty][tx] = 0.0;
	if (j+TILE_SIZE<Nj && ty<Nk)	bs1[0][ty][tx] = B[ty*Nj + j+TILE_SIZE];
	else							bs1[0][ty][tx] = 0.0;
	__syncthreads();
	
	for (int kt=0; kt<Nk; kt+=TILE_SIZE) {
		if (i<Ni && j<Nj) {
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum00 += as0[current][k-kt][ty]*bs0[current][k-kt][tx];
		}
		if (i<Ni && j+TILE_SIZE<Nj) {
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum01 += as0[current][k-kt][ty]*bs1[current][k-kt][tx];
		}
		if (i+TILE_SIZE<Ni && j<Nj) {
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum10 += as1[current][k-kt][ty]*bs0[current][k-kt][tx];
		}
		if (i+TILE_SIZE<Ni && j+TILE_SIZE<Nj) {
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum11 += as1[current][k-kt][ty]*bs1[current][k-kt][tx];
		}
		current = (current+1)&1;
		if (i<Ni && (kt+TILE_SIZE+tx)<Nk)
			as0[current][tx][ty] = A[(kt+TILE_SIZE+tx)*Ni + i];
		else
			as0[current][tx][ty] = 0.0;
		if (i+TILE_SIZE<Ni && (kt+TILE_SIZE+tx)<Nk)
			as1[current][tx][ty] = A[(kt+TILE_SIZE+tx)*Ni + i+TILE_SIZE];
		else
			as1[current][tx][ty] = 0.0;
		if (j<Nj && (kt+TILE_SIZE+ty)<Nk)
			bs0[current][ty][tx] = B[(kt+TILE_SIZE+ty)*Nj + j];
		else
			bs0[current][ty][tx] = 0.0;
		if (j+TILE_SIZE<Nj && (kt+TILE_SIZE+ty)<Nk)
			bs1[current][ty][tx] = B[(kt+TILE_SIZE+ty)*Nj + j+TILE_SIZE];
		else
			bs1[current][ty][tx] = 0.0;
		__syncthreads();
	}
	if (i<Ni && j<Nj)	C[i*Nj + j] = sum00;
	if (i<Ni && j+TILE_SIZE<Nj)	C[i*Nj + j+TILE_SIZE] = sum01;
	if (i+TILE_SIZE<Ni && j<Nj)	C[(i+TILE_SIZE)*Nj + j] = sum10;
	if (i+TILE_SIZE<Ni && j+TILE_SIZE<Nj)	C[(i+TILE_SIZE)*Nj + j+TILE_SIZE] = sum11;
}

__global__ void aTb_gpu_i2j2(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk)
{
	const unsigned int tx = threadIdx.x;
	const unsigned int ty = threadIdx.y;
	const unsigned int i = blockIdx.y*blockDim.y*2 + ty;
	const unsigned int j = blockIdx.x*blockDim.x*2 + tx;
	__shared__ float as0[TILE_SIZE][TILE_SIZE];
	__shared__ float as1[TILE_SIZE][TILE_SIZE];
	__shared__ float bs0[TILE_SIZE][TILE_SIZE];
	__shared__ float bs1[TILE_SIZE][TILE_SIZE];
	float sum00 = 0.0;
	float sum01 = 0.0;
	float sum10 = 0.0;
	float sum11 = 0.0;
	for (int kt=0; kt<Nk; kt+=TILE_SIZE) {
		if (i<Ni && (kt+tx)<Nk)		as0[tx][ty] = A[(kt+tx)*Ni + i];
		else						as0[tx][ty] = 0.0;
		if (i+TILE_SIZE<Ni && (kt+tx)<Nk)	as1[tx][ty] = A[(kt+tx)*Ni + i+TILE_SIZE];
		else								as1[tx][ty] = 0.0;
		if (j<Nj && (kt+ty)<Nk)		bs0[ty][tx] = B[(kt+ty)*Nj + j];
		else						bs0[ty][tx] = 0.0;
		if (j+TILE_SIZE<Nj && (kt+ty)<Nk)	bs1[ty][tx] = B[(kt+ty)*Nj + j+TILE_SIZE];
		else								bs1[ty][tx] = 0.0;
		__syncthreads();

		if (i<Ni && j<Nj) {
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum00 += as0[k-kt][ty]*bs0[k-kt][tx];
		}
		if (i<Ni && j+TILE_SIZE<Nj) {
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum01 += as0[k-kt][ty]*bs1[k-kt][tx];
		}
		if (i+TILE_SIZE<Ni && j<Nj) {
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum10 += as1[k-kt][ty]*bs0[k-kt][tx];
		}
		if (i+TILE_SIZE<Ni && j+TILE_SIZE<Nj) {
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum11 += as1[k-kt][ty]*bs1[k-kt][tx];
		}
		__syncthreads();
	}
	if (i<Ni && j<Nj)	C[i*Nj + j] = sum00;
	if (i<Ni && j+TILE_SIZE<Nj)	C[i*Nj + j+TILE_SIZE] = sum01;
	if (i+TILE_SIZE<Ni && j<Nj)	C[(i+TILE_SIZE)*Nj + j] = sum10;
	if (i+TILE_SIZE<Ni && j+TILE_SIZE<Nj)	C[(i+TILE_SIZE)*Nj + j+TILE_SIZE] = sum11;
}

__global__ void aTb_gpu_db(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk)
{
	const unsigned int tx = threadIdx.x;
	const unsigned int ty = threadIdx.y;
	const unsigned int i = blockIdx.y*blockDim.y + ty;
	const unsigned int j = blockIdx.x*blockDim.x + tx;

	__shared__ float as[2][TILE_SIZE][TILE_SIZE];
	__shared__ float bs[2][TILE_SIZE][TILE_SIZE];
	float sum = 0;
	unsigned short int current = 0;

	if (i<Ni && tx<Nk)	as[0][tx][ty] = A[tx*Ni + i];
	else				as[0][tx][ty] = 0.0;
	if (j<Nj && ty<Nk)	bs[0][ty][tx] = B[ty*Nj + j];
	else				bs[0][ty][tx] = 0.0;
	__syncthreads();

	for (int kt=0; kt<Nk; kt+=TILE_SIZE) {
		if (i<Ni && j<Nj) {
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum += as[current][k-kt][ty]*bs[current][k-kt][tx];
		}
		current = (current+1)&1;
		if (i<Ni && (kt+TILE_SIZE+tx)<Nk)	as[current][tx][ty] = A[(kt+TILE_SIZE+tx)*Ni + i];
		else								as[current][tx][ty] = 0.0;
		if (j<Nj && (kt+TILE_SIZE+ty)<Nk)	bs[current][ty][tx] = B[(kt+TILE_SIZE+ty)*Nj + j];
		else								bs[current][ty][tx] = 0.0;
		__syncthreads();
	}
	if (i<Ni && j<Nj)	C[i*Nj + j] = sum;
}

__global__ void abT_gpu(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk)
{
  int i, j, k;

  for (i = 0; i < Ni; i++) 
   for (j = 0; j < Nj; j++)
    C[i*Nj+j]=0.0;
  for (i = 0; i < Ni; i++)
   for (j = 0; j < Nj; j++)
    for (k = 0; k < Nk; k++)
// C[i][j] = C[i][j] + A[i][k]*B[j][k];
     C[i*Nj+j]=C[i*Nj+j]+A[i*Nk+k]*B[j*Nk+k];
}
// abT i4j4
__global__ void abT_gpu_i4j4(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk)
{
	const unsigned tx = threadIdx.x;
	const unsigned ty = threadIdx.y;
	const unsigned j = blockIdx.x*blockDim.x*4 + tx;
	const unsigned i = blockIdx.y*blockDim.y*4 + ty;
	float sum0000 = 0.0;
	float sum0001 = 0.0;
	float sum0010 = 0.0;
	float sum0011 = 0.0;
	float sum0100 = 0.0;
	float sum0101 = 0.0;
	float sum0110 = 0.0;
	float sum0111 = 0.0;
	float sum1000 = 0.0;
	float sum1001 = 0.0;
	float sum1010 = 0.0;
	float sum1011 = 0.0;
	float sum1100 = 0.0;
	float sum1101 = 0.0;
	float sum1110 = 0.0;
	float sum1111 = 0.0;

	__shared__ float as00[TILE_SIZE][TILE_SIZE];
	__shared__ float as01[TILE_SIZE][TILE_SIZE];
	__shared__ float as10[TILE_SIZE][TILE_SIZE];
	__shared__ float as11[TILE_SIZE][TILE_SIZE];
	__shared__ float bs00[TILE_SIZE][TILE_SIZE];
	__shared__ float bs01[TILE_SIZE][TILE_SIZE];
	__shared__ float bs10[TILE_SIZE][TILE_SIZE];
	__shared__ float bs11[TILE_SIZE][TILE_SIZE];

	for (int kt=0; kt<Nk; kt+=TILE_SIZE) {
		if (i<Ni && (kt+tx)<Nk)		as00[ty][tx] = A[i*Nk + kt+tx];
		else						as00[ty][tx] = 0.0;
		if (i+TILE_SIZE<Ni && (kt+tx)<Nk)	as01[ty][tx] = A[(i+TILE_SIZE)*Nk + kt+tx];
		else								as01[ty][tx] = 0.0;
		if (i+2*TILE_SIZE<Ni && (kt+tx)<Nk)	as10[ty][tx] = A[(i+2*TILE_SIZE)*Nk + kt+tx];
		else								as10[ty][tx] = 0.0;
		if (i+3*TILE_SIZE<Ni && (kt+tx)<Nk)	as11[ty][tx] = A[(i+3*TILE_SIZE)*Nk + kt+tx];
		else	as11[ty][tx] = 0.0;

		if (j<Nj && (kt+ty)<Nk)		bs00[tx][ty] = B[j*Nk + kt+ty];
		else						bs00[tx][ty] = 0.0;
		if (j+TILE_SIZE<Nj && (kt+ty)<Nk)	bs01[tx][ty] = B[(j+TILE_SIZE)*Nk + kt+ty];
		else								bs01[tx][ty] = 0.0;
		if (j+2*TILE_SIZE<Nj && (kt+ty)<Nk)	bs10[tx][ty] = B[(j+2*TILE_SIZE)*Nk + kt+ty];
		else								bs10[tx][ty] = 0.0;
		if (j+3*TILE_SIZE<Nj && (kt+ty)<Nk)	bs11[tx][ty] = B[(j+3*TILE_SIZE)*Nk + kt+ty];
		else								bs11[tx][ty] = 0.0;
		__syncthreads();

		if (i<Ni && j<Nj)
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum0000 += as00[ty][k-kt]*bs00[tx][k-kt];
		if (i<Ni && j+TILE_SIZE<Nj)
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum0001 += as00[ty][k-kt]*bs01[tx][k-kt];
		if (i<Ni && j+2*TILE_SIZE<Nj)
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum0010 += as00[ty][k-kt]*bs10[tx][k-kt];
		if (i<Ni && j+3*TILE_SIZE<Nj)
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum0011 += as00[ty][k-kt]*bs11[tx][k-kt];
		if (i+TILE_SIZE<Ni && j<Nj)
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum0100 += as01[ty][k-kt]*bs00[tx][k-kt];
		if (i+TILE_SIZE<Ni && j+TILE_SIZE<Nj)
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum0101 += as01[ty][k-kt]*bs01[tx][k-kt];
		if (i+TILE_SIZE<Ni && j+2*TILE_SIZE<Nj)
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum0110 += as01[ty][k-kt]*bs10[tx][k-kt];
		if (i+TILE_SIZE<Ni && j+3*TILE_SIZE<Nj)
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum0111 += as01[ty][k-kt]*bs11[tx][k-kt];
		if (i+2*TILE_SIZE<Ni && j<Nj)
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum1000 += as10[ty][k-kt]*bs00[tx][k-kt];
		if (i+2*TILE_SIZE<Ni && j+TILE_SIZE<Nj)
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum1001 += as10[ty][k-kt]*bs01[tx][k-kt];
		if (i+2*TILE_SIZE<Ni && j+2*TILE_SIZE<Nj)
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum1010 += as10[ty][k-kt]*bs10[tx][k-kt];
		if (i+2*TILE_SIZE<Ni && j+3*TILE_SIZE<Nj)
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum1011 += as10[ty][k-kt]*bs11[tx][k-kt];
		if (i+3*TILE_SIZE<Ni && j<Nj)
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum1100 += as11[ty][k-kt]*bs00[tx][k-kt];
		if (i+3*TILE_SIZE<Ni && j+TILE_SIZE<Nj)
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum1101 += as11[ty][k-kt]*bs01[tx][k-kt];
		if (i+3*TILE_SIZE<Ni && j+2*TILE_SIZE<Nj)
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum1110 += as11[ty][k-kt]*bs10[tx][k-kt];
		if (i+3*TILE_SIZE<Ni && j+3*TILE_SIZE<Nj)
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum1111 += as11[ty][k-kt]*bs11[tx][k-kt];
		__syncthreads();
	}
	if (i<Ni && j<Nj)				C[i*Nj + j] = sum0000;
	if (i<Ni && j+TILE_SIZE<Nj)		C[i*Nj + j+TILE_SIZE] = sum0001;
	if (i<Ni && j+2*TILE_SIZE<Nj)	C[i*Nj + j+2*TILE_SIZE] = sum0010;
	if (i<Ni && j+3*TILE_SIZE<Nj)	C[i*Nj + j+3*TILE_SIZE] = sum0011;
	if (i+TILE_SIZE<Ni && j<Nj)				C[(i+TILE_SIZE)*Nj + j] = sum0100;
	if (i+TILE_SIZE<Ni && j+TILE_SIZE<Nj)	C[(i+TILE_SIZE)*Nj + j+TILE_SIZE] = sum0101;
	if (i+TILE_SIZE<Ni && j+2*TILE_SIZE<Nj)	C[(i+TILE_SIZE)*Nj + j+2*TILE_SIZE] = sum0110;
	if (i+TILE_SIZE<Ni && j+3*TILE_SIZE<Nj)	C[(i+TILE_SIZE)*Nj + j+3*TILE_SIZE] = sum0111;
	if (i+2*TILE_SIZE<Ni && j<Nj)				C[(i+2*TILE_SIZE)*Nj + j] = sum1000;
	if (i+2*TILE_SIZE<Ni && j+TILE_SIZE<Nj)		C[(i+2*TILE_SIZE)*Nj + j+TILE_SIZE] = sum1001;
	if (i+2*TILE_SIZE<Ni && j+2*TILE_SIZE<Nj)	C[(i+2*TILE_SIZE)*Nj + j+2*TILE_SIZE] = sum1010;
	if (i+2*TILE_SIZE<Ni && j+3*TILE_SIZE<Nj)	C[(i+2*TILE_SIZE)*Nj + j+3*TILE_SIZE] = sum1011;
	if (i+3*TILE_SIZE<Ni && j<Nj)				C[(i+3*TILE_SIZE)*Nj + j] = sum1100;
	if (i+3*TILE_SIZE<Ni && j+TILE_SIZE<Nj)		C[(i+3*TILE_SIZE)*Nj + j+TILE_SIZE] = sum1101;
	if (i+3*TILE_SIZE<Ni && j+2*TILE_SIZE<Nj)	C[(i+3*TILE_SIZE)*Nj + j+2*TILE_SIZE] = sum1110;
	if (i+3*TILE_SIZE<Ni && j+3*TILE_SIZE<Nj)	C[(i+3*TILE_SIZE)*Nj + j+3*TILE_SIZE] = sum1111;
}

__global__ void abT_gpu_i2j2db(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk)
{
	const unsigned tx = threadIdx.x;
	const unsigned ty = threadIdx.y;
	const unsigned j = blockIdx.x*blockDim.x*2 + tx;
	const unsigned i = blockIdx.y*blockDim.y*2 + ty;
	float sum00 = 0.0;
	float sum01 = 0.0;
	float sum10 = 0.0;
	float sum11 = 0.0;
	unsigned short int current = 0;
	__shared__ float as0[2][TILE_SIZE][TILE_SIZE];
	__shared__ float as1[2][TILE_SIZE][TILE_SIZE];
	__shared__ float bs0[2][TILE_SIZE][TILE_SIZE];
	__shared__ float bs1[2][TILE_SIZE][TILE_SIZE];
	if (i<Ni && tx<Nk)	as0[0][ty][tx] = A[i*Nk + tx];
	else				as0[0][ty][tx] = 0.0;
	if (i+TILE_SIZE<Ni && tx<Nk)	as1[0][ty][tx] = A[(i+TILE_SIZE)*Nk + tx];
	else							as1[0][ty][tx] = 0.0;
	if (j<Nj && ty<Nk)	bs0[0][tx][ty] = B[j*Nk + ty];
	else				bs0[0][tx][ty] = 0.0;
	if (j+TILE_SIZE<Nj && ty<Nk)	bs1[0][tx][ty] = B[(j+TILE_SIZE)*Nk + ty];
	else							bs1[0][tx][ty] = 0.0;
	__syncthreads();

	for (int kt=0; kt<Nk; kt+=TILE_SIZE) {
		if (i<Ni && j<Nj)
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum00 += as0[current][ty][k-kt]*bs0[current][tx][k-kt];
		if (i<Ni && j+TILE_SIZE<Nj)
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum01 += as0[current][ty][k-kt]*bs1[current][tx][k-kt];
		if (i+TILE_SIZE<Ni && j<Nj)
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum10 += as1[current][ty][k-kt]*bs0[current][tx][k-kt];
		if (i<Ni && j+TILE_SIZE<Nj)
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum11 += as1[current][ty][k-kt]*bs1[current][tx][k-kt];

		current = (current+1)&1;
		if (i<Ni && (kt+TILE_SIZE+tx)<Nk)	as0[current][ty][tx] = A[i*Nk + kt+TILE_SIZE+tx];
		else								as0[current][ty][tx] = 0.0;
		if (i+TILE_SIZE<Ni && (kt+TILE_SIZE+tx)<Nk)	as1[current][ty][tx] = A[(i+TILE_SIZE)*Nk + kt+TILE_SIZE+tx];
		else										as1[current][ty][tx] = 0.0;
		if (j<Nj && (kt+TILE_SIZE+ty)<Nk)	bs0[current][tx][ty] = B[j*Nk + kt+TILE_SIZE+ty];
		else								bs0[current][tx][ty] = 0.0;
		if (j+TILE_SIZE<Nj && (kt+TILE_SIZE+ty)<Nk)	bs1[current][tx][ty] = B[(j+TILE_SIZE)*Nk + kt+TILE_SIZE+ty];
		else										bs1[current][tx][ty] = 0.0;
		__syncthreads();
	}
	if (i<Ni && j<Nj)	C[i*Nj + j] = sum00;
	if (i<Ni && j+TILE_SIZE<Nj)	C[i*Nj + j+TILE_SIZE] = sum01;
	if (i+TILE_SIZE<Ni && j<Nj)	C[(i+TILE_SIZE)*Nj + j] = sum10;
	if (i+TILE_SIZE<Ni && j+TILE_SIZE<Nj)	C[(i+TILE_SIZE)*Nj + j+TILE_SIZE] = sum11;
}

__global__ void abT_gpu_db(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk)
{
	const unsigned tx = threadIdx.x;
	const unsigned ty = threadIdx.y;
	const unsigned j = blockIdx.x*blockDim.x + tx;
	const unsigned i = blockIdx.y*blockDim.y + ty;
	float sum = 0;
	unsigned short int current = 0;
	__shared__ float as[2][TILE_SIZE][TILE_SIZE];
	__shared__ float bs[2][TILE_SIZE][TILE_SIZE];
	if (i<Ni && tx<Nk)	as[0][ty][tx] = A[i*Nk + tx];
	else				as[0][ty][tx] = 0.0;
	if (j<Nj && ty<Nk)	bs[0][tx][ty] = B[j*Nk + ty];
	else				bs[0][tx][ty] = 0.0;
	__syncthreads();
	for (int kt=0; kt<Nk; kt+=TILE_SIZE) {
		if (i<Ni && j<Nj)
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum += as[current][ty][k-kt]*bs[current][tx][k-kt];

		current = (current+1)&1;
		if (i<Ni && (kt+TILE_SIZE+tx)<Nk)	as[current][ty][tx] = A[i*Nk + kt+TILE_SIZE+tx];
		else								as[current][ty][tx] = 0.0;
		if (j<Nj && (kt+TILE_SIZE+ty)<Nk)	bs[current][tx][ty] = B[j*Nk + kt+TILE_SIZE+ty];
		else								bs[current][tx][ty] = 0.0;
		__syncthreads();
	}
	if (i<Ni && j<Nj)	C[i*Nj + j] = sum;
}

__global__ void aTbT_gpu(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk)
{
  int i, j, k;

  for (i = 0; i < Ni; i++) 
   for (j = 0; j < Nj; j++)
    C[i*Nj+j]=0.0;
  for (i = 0; i < Ni; i++)
   for (j = 0; j < Nj; j++)
    for (k = 0; k < Nk; k++)
// C[i][j] = C[i][j] + A[k][i]*B[j][k];
     C[i*Nj+j]=C[i*Nj+j]+A[k*Ni+i]*B[j*Nk+k];
}

__global__ void aTbT_gpu_i4j4(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk)
{
	const unsigned int tx = threadIdx.x;
	const unsigned int ty = threadIdx.y;
	const unsigned int j = blockIdx.x*blockDim.x*4 + tx;
	const unsigned int i = blockIdx.y*blockDim.y*4 + ty;
	float sum0000 = 0.0;
	float sum0001 = 0.0;
	float sum0010 = 0.0;
	float sum0011 = 0.0;
	float sum0100 = 0.0;
	float sum0101 = 0.0;
	float sum0110 = 0.0;
	float sum0111 = 0.0;
	float sum1000 = 0.0;
	float sum1001 = 0.0;
	float sum1010 = 0.0;
	float sum1011 = 0.0;
	float sum1100 = 0.0;
	float sum1101 = 0.0;
	float sum1110 = 0.0;
	float sum1111 = 0.0;
	__shared__ float as00[TILE_SIZE][TILE_SIZE];
	__shared__ float as01[TILE_SIZE][TILE_SIZE];
	__shared__ float as10[TILE_SIZE][TILE_SIZE];
	__shared__ float as11[TILE_SIZE][TILE_SIZE];
	__shared__ float bs00[TILE_SIZE][TILE_SIZE];
	__shared__ float bs01[TILE_SIZE][TILE_SIZE];
	__shared__ float bs10[TILE_SIZE][TILE_SIZE];
	__shared__ float bs11[TILE_SIZE][TILE_SIZE];
	for (int kt=0; kt<Nk; kt+=TILE_SIZE) {
		if (i<Ni && (kt+tx)<Nk)				as00[tx][ty] = A[(kt+tx)*Ni + i];
		else								as00[tx][ty] = 0.0;
		if (i+TILE_SIZE<Ni && (kt+tx)<Nk)	as01[tx][ty] = A[(kt+tx)*Ni + i+TILE_SIZE];
		else								as01[tx][ty] = 0.0;
		if (i+2*TILE_SIZE<Ni && (kt+tx)<Nk)	as10[tx][ty] = A[(kt+tx)*Ni + i+2*TILE_SIZE];
		else								as10[tx][ty] = 0.0;
		if (i+3*TILE_SIZE<Ni && (kt+tx)<Nk)	as11[tx][ty] = A[(kt+tx)*Ni + i+3*TILE_SIZE];
		else								as11[tx][ty] = 0.0;
		if (j<Nj && (kt+ty)<Nk)				bs00[tx][ty] = B[j*Nk + kt+ty];
		else								bs00[tx][ty] = 0.0;
		if (j+TILE_SIZE<Nj && (kt+ty)<Nk)	bs01[tx][ty] = B[(j+TILE_SIZE)*Nk + kt+ty];
		else								bs01[tx][ty] = 0.0;
		if (j+2*TILE_SIZE<Nj && (kt+ty)<Nk)	bs10[tx][ty] = B[(j+2*TILE_SIZE)*Nk + kt+ty];
		else								bs10[tx][ty] = 0.0;
		if (j+3*TILE_SIZE<Nj && (kt+ty)<Nk)	bs11[tx][ty] = B[(j+3*TILE_SIZE)*Nk + kt+ty];
		else								bs11[tx][ty] = 0.0;
		__syncthreads();
		if (i<Ni && j<Nj)
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum0000 += as00[k-kt][ty]*bs00[tx][k-kt];
		if (i<Ni && j+TILE_SIZE<Nj)
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum0001 += as00[k-kt][ty]*bs01[tx][k-kt];
		if (i<Ni && j+2*TILE_SIZE<Nj)
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum0010 += as00[k-kt][ty]*bs10[tx][k-kt];
		if (i<Ni && j+3*TILE_SIZE<Nj)
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum0011 += as00[k-kt][ty]*bs11[tx][k-kt];
		if (i+TILE_SIZE<Ni && j<Nj)
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum0100 += as01[k-kt][ty]*bs00[tx][k-kt];
		if (i+TILE_SIZE<Ni && j+TILE_SIZE<Nj)
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum0101 += as01[k-kt][ty]*bs01[tx][k-kt];
		if (i+TILE_SIZE<Ni && j+2*TILE_SIZE<Nj)
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum0110 += as01[k-kt][ty]*bs10[tx][k-kt];
		if (i+TILE_SIZE<Ni && j+3*TILE_SIZE<Nj)
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum0111 += as01[k-kt][ty]*bs11[tx][k-kt];
		if (i+2*TILE_SIZE<Ni && j<Nj)
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum1000 += as10[k-kt][ty]*bs00[tx][k-kt];
		if (i+2*TILE_SIZE<Ni && j+TILE_SIZE<Nj)
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum1001 += as10[k-kt][ty]*bs01[tx][k-kt];
		if (i+2*TILE_SIZE<Ni && j+2*TILE_SIZE<Nj)
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum1010 += as10[k-kt][ty]*bs10[tx][k-kt];
		if (i+2*TILE_SIZE<Ni && j+3*TILE_SIZE<Nj)
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum1011 += as10[k-kt][ty]*bs11[tx][k-kt];
		if (i+3*TILE_SIZE<Ni && j<Nj)
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum1100 += as11[k-kt][ty]*bs00[tx][k-kt];
		if (i+3*TILE_SIZE<Ni && j+TILE_SIZE<Nj)
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum1101 += as11[k-kt][ty]*bs01[tx][k-kt];
		if (i+3*TILE_SIZE<Ni && j+2*TILE_SIZE<Nj)
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum1110 += as11[k-kt][ty]*bs10[tx][k-kt];
		if (i+3*TILE_SIZE<Ni && j+3*TILE_SIZE<Nj)
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum1111 += as11[k-kt][ty]*bs11[tx][k-kt];
		__syncthreads();
	}
	if (i<Ni && j<Nj)				C[i*Nj + j] = sum0000;
	if (i<Ni && j+TILE_SIZE<Nj)		C[i*Nj + j+TILE_SIZE] = sum0001;
	if (i<Ni && j+2*TILE_SIZE<Nj)	C[i*Nj + j+2*TILE_SIZE] = sum0010;
	if (i<Ni && j+3*TILE_SIZE<Nj)	C[i*Nj + j+3*TILE_SIZE] = sum0011;
	if (i+TILE_SIZE<Ni && j<Nj)				C[(i+TILE_SIZE)*Nj + j] = sum0100;
	if (i+TILE_SIZE<Ni && j+TILE_SIZE<Nj)	C[(i+TILE_SIZE)*Nj + j+TILE_SIZE] = sum0101;
	if (i+TILE_SIZE<Ni && j+2*TILE_SIZE<Nj)	C[(i+TILE_SIZE)*Nj + j+2*TILE_SIZE] = sum0110;
	if (i+TILE_SIZE<Ni && j+3*TILE_SIZE<Nj)	C[(i+TILE_SIZE)*Nj + j+3*TILE_SIZE] = sum0111;
	if (i+2*TILE_SIZE<Ni && j<Nj)				C[(i+2*TILE_SIZE)*Nj + j] = sum1000;
	if (i+2*TILE_SIZE<Ni && j+TILE_SIZE<Nj)		C[(i+2*TILE_SIZE)*Nj + j+TILE_SIZE] = sum1001;
	if (i+2*TILE_SIZE<Ni && j+2*TILE_SIZE<Nj)	C[(i+2*TILE_SIZE)*Nj + j+2*TILE_SIZE] = sum1010;
	if (i+2*TILE_SIZE<Ni && j+3*TILE_SIZE<Nj)	C[(i+2*TILE_SIZE)*Nj + j+3*TILE_SIZE] = sum1011;
	if (i+3*TILE_SIZE<Ni && j<Nj)				C[(i+3*TILE_SIZE)*Nj + j] = sum1100;
	if (i+3*TILE_SIZE<Ni && j+TILE_SIZE<Nj)		C[(i+3*TILE_SIZE)*Nj + j+TILE_SIZE] = sum1101;
	if (i+3*TILE_SIZE<Ni && j+2*TILE_SIZE<Nj)	C[(i+3*TILE_SIZE)*Nj + j+2*TILE_SIZE] = sum1110;
	if (i+3*TILE_SIZE<Ni && j+3*TILE_SIZE<Nj)	C[(i+3*TILE_SIZE)*Nj + j+3*TILE_SIZE] = sum1111;
}

__global__ void aTbT_gpu_i2j2(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk)
{
	const unsigned int tx = threadIdx.x;
	const unsigned int ty = threadIdx.y;
	const unsigned int j = blockIdx.x*blockDim.x*2 + tx;
	const unsigned int i = blockIdx.y*blockDim.y*2 + ty;
	float sum00 = 0.0;
	float sum01 = 0.0;
	float sum10 = 0.0;
	float sum11 = 0.0;
	__shared__ float as0[TILE_SIZE][TILE_SIZE];
	__shared__ float as1[TILE_SIZE][TILE_SIZE];
	__shared__ float bs0[TILE_SIZE][TILE_SIZE];
	__shared__ float bs1[TILE_SIZE][TILE_SIZE];
	for (int kt=0; kt<Nk; kt+=TILE_SIZE) {
		if (i<Ni && (kt+tx)<Nk)				as0[tx][ty] = A[(kt+tx)*Ni + i];
		else								as0[tx][ty] = 0.0;
		if (i+TILE_SIZE<Ni && (kt+tx)<Nk)	as1[tx][ty] = A[(kt+tx)*Ni + i+TILE_SIZE];
		else								as1[tx][ty] = 0.0;
		if (j<Nj && (kt+ty)<Nk)				bs0[tx][ty] = B[j*Nk + kt+ty];
		else								bs0[tx][ty] = 0.0;
		if (j+TILE_SIZE<Nj && (kt+ty)<Nk)	bs1[tx][ty] = B[(j+TILE_SIZE)*Nk + kt+ty];
		else								bs1[tx][ty] = 0.0;
		__syncthreads();
		if (i<Ni && j<Nj)
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum00 += as0[k-kt][ty]*bs0[tx][k-kt];
		if (i<Ni && j+TILE_SIZE<Nj)
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum01 += as0[k-kt][ty]*bs1[tx][k-kt];
		if (i+TILE_SIZE<Ni && j<Nj)
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum10 += as1[k-kt][ty]*bs0[tx][k-kt];
		if (i+TILE_SIZE<Ni && j+TILE_SIZE<Nj)
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum11 += as1[k-kt][ty]*bs1[tx][k-kt];
		__syncthreads();
	}
	if (i<Ni && j<Nj)			C[i*Nj + j] = sum00;
	if (i<Ni && j+TILE_SIZE<Nj)	C[i*Nj + j+TILE_SIZE] = sum01;
	if (i+TILE_SIZE<Ni && j<Nj)	C[(i+TILE_SIZE)*Nj + j] = sum10;
	if (i+TILE_SIZE<Ni && j+TILE_SIZE<Nj)	C[(i+TILE_SIZE)*Nj + j+TILE_SIZE] = sum11;
}

__global__ void aTbT_gpu_db(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk)
{
	const unsigned int tx = threadIdx.x;
	const unsigned int ty = threadIdx.y;
	const unsigned int j = blockIdx.x*blockDim.x + tx;
	const unsigned int i = blockIdx.y*blockDim.y + ty;
	float sum = 0;
	unsigned short int current = 0;
	__shared__ float as[2][TILE_SIZE][TILE_SIZE];
	__shared__ float bs[2][TILE_SIZE][TILE_SIZE];
	if (i<Ni && tx<Nk)	as[0][tx][ty] = A[tx*Ni + i];
	else				as[0][tx][ty] = 0.0;
	if (j<Nj && ty<Nk)	bs[0][tx][ty] = B[j*Nk + ty];
	else				bs[0][tx][ty] = 0.0;
	__syncthreads();
	for (int kt=0; kt<Nk; kt+=TILE_SIZE) {
		if (i<Ni && j<Nj)
			for (int k=kt; k<MIN(kt+TILE_SIZE,Nk); k++)
				sum += as[current][k-kt][ty]*bs[current][tx][k-kt];

		current = (current+1)&1;
		if (i<Ni && (kt+TILE_SIZE+tx)<Nk)	as[current][tx][ty] = A[(kt+TILE_SIZE+tx)*Ni + i];
		else								as[current][tx][ty] = 0.0;
		if (j<Nj && (kt+TILE_SIZE+ty)<Nk)	bs[current][tx][ty] = B[j*Nk + kt+TILE_SIZE+ty];
		else								bs[current][tx][ty] = 0.0;
		__syncthreads();
	}
	if (i<Ni && j<Nj)	C[i*Nj + j] = sum;
}

