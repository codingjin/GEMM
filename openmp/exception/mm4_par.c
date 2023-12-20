
void ab_par(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk)
{
	int i, j, k;
	//i-2unroll and k-2unroll
	int remi = Ni&1;
	if (remi) {
		for (k=0; k<Nk; k++)
			for (j=0; j<Nj; j++)
				C[j] += A[k]*B[k*Nj+j];
	}
	#pragma omp parallel for private(i, k, j)
	for (i=remi; i<Ni; i+=2) {
		int remk = Nk&1;
		if (remk) {
			for (j=0; j<Nj; j++) {
				C[i*Nj+j] += A[i*Nk]*B[j];
				C[(i+1)*Nj+j] += A[(i+1)*Nk]*B[j];
			}
		}

		for (k=remk; k<Nk; k+=2) {
			for (j=0; j<Nj; j++) {
				C[i*Nj+j] = C[i*Nj+j] + A[i*Nk+k]*B[k*Nj+j] + A[i*Nk+k+1]*B[(k+1)*Nj+j];
				C[(i+1)*Nj+j] = C[(i+1)*Nj+j] + A[(i+1)*Nk+k]*B[k*Nj+j] + A[(i+1)*Nk+k+1]*B[(k+1)*Nj+j];
			}
		}
	}
}

void ab_par00(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk) //ik
{
	int i, j, k;
	#pragma omp parallel for private(i, k, j)
	for (i=0; i<Ni; i+=2) {
		for (k=0; k<Nk; k+=2) {
			for (j=0; j<Nj; j++) {
				C[i*Nj+j] = C[i*Nj+j] + A[i*Nk+k]*B[k*Nj+j] + A[i*Nk+k+1]*B[(k+1)*Nj+j];
				C[(i+1)*Nj+j] = C[(i+1)*Nj+j] + A[(i+1)*Nk+k]*B[k*Nj+j] + A[(i+1)*Nk+k+1]*B[(k+1)*Nj+j];
			}
		}
	}
}

void ab_par01(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk) //ik
{
	int i, j, k;
	//i-2unroll and k-2unroll
	#pragma omp parallel for private(i, k, j)
	for (i=0; i<Ni; i+=2) {
		for (j=0; j<Nj; j++) {
			C[i*Nj+j] += A[i*Nk]*B[j];
			C[(i+1)*Nj+j] += A[(i+1)*Nk]*B[j];
		}

		for (k=1; k<Nk; k+=2) {
			for (j=0; j<Nj; j++) {
				C[i*Nj+j] = C[i*Nj+j] + A[i*Nk+k]*B[k*Nj+j] + A[i*Nk+k+1]*B[(k+1)*Nj+j];
				C[(i+1)*Nj+j] = C[(i+1)*Nj+j] + A[(i+1)*Nk+k]*B[k*Nj+j] + A[(i+1)*Nk+k+1]*B[(k+1)*Nj+j];
			}
		}
	}
}

void ab_par10(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk) //ij
{
	int i, j, k;
	//i-2unroll and k-2unroll
	for (k=0; k<Nk; k++)
		for (j=0; j<Nj; j++)
			C[j] += A[k]*B[k*Nj+j];
	#pragma omp parallel for private(i, k, j)
	for (i=1; i<Ni; i+=2) {
		for (k=0; k<Nk; k+=2) {
			for (j=0; j<Nj; j++) {
				C[i*Nj+j] = C[i*Nj+j] + A[i*Nk+k]*B[k*Nj+j] + A[i*Nk+k+1]*B[(k+1)*Nj+j];
				C[(i+1)*Nj+j] = C[(i+1)*Nj+j] + A[(i+1)*Nk+k]*B[k*Nj+j] + A[(i+1)*Nk+k+1]*B[(k+1)*Nj+j];
			}
		}
	}
}

void ab_par11(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk) //ij
{
	int i, j, k;
	//i-2unroll and k-2unroll
	for (k=0; k<Nk; k++)
		for (j=0; j<Nj; j++)
			C[j] += A[k]*B[k*Nj+j];
	#pragma omp parallel for private(i, k, j)
	for (i=1; i<Ni; i+=2) {
		for (j=0; j<Nj; j++) {
			C[i*Nj+j] += A[i*Nk]*B[j];
			C[(i+1)*Nj+j] += A[(i+1)*Nk]*B[j];
		}

		for (k=1; k<Nk; k+=2) {
			for (j=0; j<Nj; j++) {
				C[i*Nj+j] = C[i*Nj+j] + A[i*Nk+k]*B[k*Nj+j] + A[i*Nk+k+1]*B[(k+1)*Nj+j];
				C[(i+1)*Nj+j] = C[(i+1)*Nj+j] + A[(i+1)*Nk+k]*B[k*Nj+j] + A[(i+1)*Nk+k+1]*B[(k+1)*Nj+j];
			}
		}
	}
}

void abT_par(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk) //ijk
{
	int i, j, k;
	int remi = Ni&1;
	if (remi)
		for (j = 0; j < Nj; j++)
			for (k = 0; k < Nk; k++)
				C[j] += A[k]*B[j*Nk+k];

	#pragma omp parallel for private(i, j, k)
	for (i=remi; i<Ni; i+=2) {
		int remj = Nj&1;
		if (remj)
			for (k = 0; k < Nk; k++) {
				C[i*Nj] += A[i*Nk+k]*B[k];
				C[(i+1)*Nj] += A[(i+1)*Nk+k]*B[k];
			}

		for (j = remj; j < Nj; j+=2)
			for (k = 0; k < Nk; k++) {
				C[i*Nj+j] = C[i*Nj+j] + A[i*Nk+k]*B[j*Nk+k];
				C[i*Nj+j+1] = C[i*Nj+j+1] + A[i*Nk+k]*B[(j+1)*Nk+k];
				C[(i+1)*Nj+j] = C[(i+1)*Nj+j] + A[(i+1)*Nk+k]*B[j*Nk+k];
				C[(i+1)*Nj+j+1] = C[(i+1)*Nj+j+1] + A[(i+1)*Nk+k]*B[(j+1)*Nk+k];
			}
	}
}

void abT_par00(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk) //ijk, ij00
{
	int i, j, k;
	#pragma omp parallel for private(i, j, k)
	for (i=0; i<Ni; i+=2) {
		for (j = 0; j < Nj; j+=2)
			for (k = 0; k < Nk; k++) {
				C[i*Nj+j] = C[i*Nj+j] + A[i*Nk+k]*B[j*Nk+k];
				C[i*Nj+j+1] = C[i*Nj+j+1] + A[i*Nk+k]*B[(j+1)*Nk+k];
				C[(i+1)*Nj+j] = C[(i+1)*Nj+j] + A[(i+1)*Nk+k]*B[j*Nk+k];
				C[(i+1)*Nj+j+1] = C[(i+1)*Nj+j+1] + A[(i+1)*Nk+k]*B[(j+1)*Nk+k];
			}
	}
}

void abT_par01(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk) //ijk ij01
{
	int i, j, k;
	#pragma omp parallel for private(i, j, k)
	for (i=0; i<Ni; i+=2) {
		for (k = 0; k < Nk; k++) {
			C[i*Nj] += A[i*Nk+k]*B[k];
			C[(i+1)*Nj] += A[(i+1)*Nk+k]*B[k];
		}

		for (j=1; j<Nj; j+=2)
			for (k = 0; k < Nk; k++) {
				C[i*Nj+j] = C[i*Nj+j] + A[i*Nk+k]*B[j*Nk+k];
				C[i*Nj+j+1] = C[i*Nj+j+1] + A[i*Nk+k]*B[(j+1)*Nk+k];
				C[(i+1)*Nj+j] = C[(i+1)*Nj+j] + A[(i+1)*Nk+k]*B[j*Nk+k];
				C[(i+1)*Nj+j+1] = C[(i+1)*Nj+j+1] + A[(i+1)*Nk+k]*B[(j+1)*Nk+k];
			}
	}
}

void abT_par10(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk) //ijk ij10
{
	int i, j, k;
	for (j = 0; j < Nj; j++)
		for (k = 0; k < Nk; k++)
			C[j] += A[k]*B[j*Nk+k];

	#pragma omp parallel for private(i, j, k)
	for (i=1; i<Ni; i+=2) {
		for (j=0; j<Nj; j+=2)
			for (k = 0; k < Nk; k++) {
				C[i*Nj+j] = C[i*Nj+j] + A[i*Nk+k]*B[j*Nk+k];
				C[i*Nj+j+1] = C[i*Nj+j+1] + A[i*Nk+k]*B[(j+1)*Nk+k];
				C[(i+1)*Nj+j] = C[(i+1)*Nj+j] + A[(i+1)*Nk+k]*B[j*Nk+k];
				C[(i+1)*Nj+j+1] = C[(i+1)*Nj+j+1] + A[(i+1)*Nk+k]*B[(j+1)*Nk+k];
			}
	}
}

void abT_par11(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk) //ijk ij11
{
	int i, j, k;
	for (j = 0; j < Nj; j++)
		for (k = 0; k < Nk; k++)
			C[j] += A[k]*B[j*Nk+k];

	#pragma omp parallel for private(i, j, k)
	for (i=1; i<Ni; i+=2) {
		for (k = 0; k < Nk; k++) {
			C[i*Nj] += A[i*Nk+k]*B[k];
			C[(i+1)*Nj] += A[(i+1)*Nk+k]*B[k];
		}

		for (j=1; j<Nj; j+=2)
			for (k = 0; k < Nk; k++) {
				C[i*Nj+j] = C[i*Nj+j] + A[i*Nk+k]*B[j*Nk+k];
				C[i*Nj+j+1] = C[i*Nj+j+1] + A[i*Nk+k]*B[(j+1)*Nk+k];
				C[(i+1)*Nj+j] = C[(i+1)*Nj+j] + A[(i+1)*Nk+k]*B[j*Nk+k];
				C[(i+1)*Nj+j+1] = C[(i+1)*Nj+j+1] + A[(i+1)*Nk+k]*B[(j+1)*Nk+k];
			}
	}
}

void aTb_par(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk)
{
	int i, j, k;
	int remi = Ni&1;
	if (remi)
		for (k=0; k<Nk; k++)
			for (j=0; j<Nj; j++)
				C[j] += A[k*Ni]*B[k*Nj+j];

	#pragma omp parallel for private(i, k, j)
	for (i=remi; i<Ni; i+=2) {
		int remk = Nk&1;
		if (remk)
			for (j=0; j<Nj; j++) {
				C[i*Nj+j] += A[i]*B[j];
				C[(i+1)*Nj+j] += A[i+1]*B[j];
			}

		for (k=remk; k<Nk; k+=2)
			for (j=0; j<Nj; j++) {
				C[i*Nj+j] = C[i*Nj+j] + A[k*Ni+i]*B[k*Nj+j] + A[(k+1)*Ni+i]*B[(k+1)*Nj+j];
				C[(i+1)*Nj+j] = C[(i+1)*Nj+j] + A[k*Ni+i+1]*B[k*Nj+j] + A[(k+1)*Ni+i+1]*B[(k+1)*Nj+j];
			}
	}
}

void aTb_par00(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk) //ik 00
{
	int i, j, k;
	#pragma omp parallel for private(i, k, j)
	for (i=0; i<Ni; i+=2) {
		for (k=0; k<Nk; k+=2)
			for (j=0; j<Nj; j++) {
				C[i*Nj+j] = C[i*Nj+j] + A[k*Ni+i]*B[k*Nj+j] + A[(k+1)*Ni+i]*B[(k+1)*Nj+j];
				C[(i+1)*Nj+j] = C[(i+1)*Nj+j] + A[k*Ni+i+1]*B[k*Nj+j] + A[(k+1)*Ni+i+1]*B[(k+1)*Nj+j];
			}
	}
}

void aTb_par01(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk) //ik 01
{
	int i, j, k;
	#pragma omp parallel for private(i, k, j)
	for (i=0; i<Ni; i+=2) {
		for (j=0; j<Nj; j++) {
			C[i*Nj+j] += A[i]*B[j];
			C[(i+1)*Nj+j] += A[i+1]*B[j];
		}

		for (k=1; k<Nk; k+=2)
			for (j=0; j<Nj; j++) {
				C[i*Nj+j] = C[i*Nj+j] + A[k*Ni+i]*B[k*Nj+j] + A[(k+1)*Ni+i]*B[(k+1)*Nj+j];
				C[(i+1)*Nj+j] = C[(i+1)*Nj+j] + A[k*Ni+i+1]*B[k*Nj+j] + A[(k+1)*Ni+i+1]*B[(k+1)*Nj+j];
			}
	}
}

void aTb_par10(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk) //ik 10
{
	int i, j, k;
	for (k=0; k<Nk; k++)
		for (j=0; j<Nj; j++)
			C[j] += A[k*Ni]*B[k*Nj+j];

	#pragma omp parallel for private(i, k, j)
	for (i=1; i<Ni; i+=2) {
		for (k=0; k<Nk; k+=2)
			for (j=0; j<Nj; j++) {
				C[i*Nj+j] = C[i*Nj+j] + A[k*Ni+i]*B[k*Nj+j] + A[(k+1)*Ni+i]*B[(k+1)*Nj+j];
				C[(i+1)*Nj+j] = C[(i+1)*Nj+j] + A[k*Ni+i+1]*B[k*Nj+j] + A[(k+1)*Ni+i+1]*B[(k+1)*Nj+j];
			}
	}
}

void aTb_par11(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk) //ik 11
{
	int i, j, k;
	for (k=0; k<Nk; k++)
		for (j=0; j<Nj; j++)
			C[j] += A[k*Ni]*B[k*Nj+j];

	#pragma omp parallel for private(i, k, j)
	for (i=1; i<Ni; i+=2) {
		for (j=0; j<Nj; j++) {
			C[i*Nj+j] += A[i]*B[j];
			C[(i+1)*Nj+j] += A[i+1]*B[j];
		}
		for (k=1; k<Nk; k+=2)
			for (j=0; j<Nj; j++) {
				C[i*Nj+j] = C[i*Nj+j] + A[k*Ni+i]*B[k*Nj+j] + A[(k+1)*Ni+i]*B[(k+1)*Nj+j];
				C[(i+1)*Nj+j] = C[(i+1)*Nj+j] + A[k*Ni+i+1]*B[k*Nj+j] + A[(k+1)*Ni+i+1]*B[(k+1)*Nj+j];
			}
	}
}

void aTbT_par(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk)
{
	int i, j, k;
	if (Ni > (Nk<<3)) { // i(2)j(2)k(2)
		int remi = Ni&1;
		if (remi)
			for (j = 0; j < Nj; j++)
				for (k = 0; k < Nk; k++)
					C[j] += A[k*Ni]*B[j*Nk+k];

		#pragma omp parallel for private(i, j, k)
		for (i=remi; i<Ni; i+=2) {
			int remj = Nj&1;
			if (remj)
				for (k=0; k<Nk; k++) {
					C[i*Nj] += A[k*Ni+i]*B[k];
					C[(i+1)*Nj] += A[k*Ni+i+1]*B[k];
				}

			for (j=remj; j<Nj; j+=2) {
				int remk = Nk&1;
				if (remk) {
					C[i*Nj+j] += A[i]*B[j*Nk];
					C[(i+1)*Nj+j] += A[i+1]*B[j*Nk];
					C[i*Nj+j+1] += A[i]*B[(j+1)*Nk];
					C[(i+1)*Nj+j+1] += A[i+1]*B[(j+1)*Nk];
				}
				for (k=remk; k<Nk; k+=2) {
					C[i*Nj+j] = C[i*Nj+j] + A[k*Ni+i]*B[j*Nk+k] + A[(k+1)*Ni+i]*B[j*Nk+k+1];
					C[(i+1)*Nj+j] = C[(i+1)*Nj+j] + A[k*Ni+i+1]*B[j*Nk+k] + A[(k+1)*Ni+i+1]*B[j*Nk+k+1];
					C[i*Nj+j+1] = C[i*Nj+j+1] + A[k*Ni+i]*B[(j+1)*Nk+k] + A[(k+1)*Ni+i]*B[(j+1)*Nk+k+1];
					C[(i+1)*Nj+j+1] = C[(i+1)*Nj+j+1] + A[k*Ni+i+1]*B[(j+1)*Nk+k] + A[(k+1)*Ni+i+1]*B[(j+1)*Nk+k+1];
				}
			}
		}
	} else if (Ni >= (Nk>>6)) { // i(2)k(2)j
		int remi = Ni&1;
		if (remi) {
			for (k = 0; k < Nk; k++)
				for (j = 0; j < Nj; j++)
					C[j] += A[k*Ni]*B[j*Nk+k];
		}
		#pragma omp parallel for private(i, j)
		for (i=remi; i<Ni; i+=2) {
			int remk = Nk&1;
			if (remk) {
				for (j=0; j<Nj; j++) {
					C[i*Nj+j] += A[i]*B[j*Nk];
					C[(i+1)*Nj+j] += A[i+1]*B[j*Nk];
				}
			}

			for (k=remk; k<Nk; k+=2) {
				for (j=0; j<Nj; j++) {
					C[i*Nj+j] += A[k*Ni+i]*B[j*Nk+k];
					C[(i+1)*Nj+j] += A[k*Ni+i+1]*B[j*Nk+k];
					C[i*Nj+j] += A[(k+1)*Ni+i]*B[j*Nk+k+1];
					C[(i+1)*Nj+j] += A[(k+1)*Ni+i+1]*B[j*Nk+k+1];
				}
			}
		}
	} else { // ikj notice, here i and j are relatively small, and k is very large
		#pragma omp parallel for private(i, j, k)
		for (i = 0; i < Ni; i++) {
			int remk = Nk&7;
			for (k=0; k<remk; k++)
				for (j=0; j<Nj; j++)
					C[i*Nj+j] += A[k*Ni+i]*B[j*Nk+k];

			for (k=remk; k<Nk; k+=8) {
					for (j = 0; j < Nj; j++)
						C[i*Nj+j] = C[i*Nj+j] + A[k*Ni+i]*B[j*Nk+k] + A[(k+1)*Ni+i]*B[j*Nk+k+1] + A[(k+2)*Ni+i]*B[j*Nk+k+2] + A[(k+3)*Ni+i]*B[j*Nk+k+3] + A[(k+4)*Ni+i]*B[j*Nk+k+4] \
									+ A[(k+5)*Ni+i]*B[j*Nk+k+5] + A[(k+6)*Ni+i]*B[j*Nk+k+6] + A[(k+7)*Ni+i]*B[j*Nk+k+7];
			}
		}
	}
}

void aTbT_par00(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk) // minn > (Nk<<3)
{
	int i, j, k;
	int remi = Ni&1;
	if (remi)
		for (j = 0; j < Nj; j++)
			for (k = 0; k < Nk; k++)
				C[j] += A[k*Ni]*B[j*Nk+k];

	#pragma omp parallel for private(i, j, k)
	for (i=remi; i<Ni; i+=2) {
		int remj = Nj&1;
		if (remj)
			for (k=0; k<Nk; k++) {
				C[i*Nj] += A[k*Ni+i]*B[k];
				C[(i+1)*Nj] += A[k*Ni+i+1]*B[k];
			}

		for (j=remj; j<Nj; j+=2) {
			int remk = Nk&1;
			if (remk) {
				C[i*Nj+j] += A[i]*B[j*Nk];
				C[(i+1)*Nj+j] += A[i+1]*B[j*Nk];
				C[i*Nj+j+1] += A[i]*B[(j+1)*Nk];
				C[(i+1)*Nj+j+1] += A[i+1]*B[(j+1)*Nk];
			}
			for (k=remk; k<Nk; k+=2) {
				C[i*Nj+j] = C[i*Nj+j] + A[k*Ni+i]*B[j*Nk+k] + A[(k+1)*Ni+i]*B[j*Nk+k+1];
				C[(i+1)*Nj+j] = C[(i+1)*Nj+j] + A[k*Ni+i+1]*B[j*Nk+k] + A[(k+1)*Ni+i+1]*B[j*Nk+k+1];
				C[i*Nj+j+1] = C[i*Nj+j+1] + A[k*Ni+i]*B[(j+1)*Nk+k] + A[(k+1)*Ni+i]*B[(j+1)*Nk+k+1];
				C[(i+1)*Nj+j+1] = C[(i+1)*Nj+j+1] + A[k*Ni+i+1]*B[(j+1)*Nk+k] + A[(k+1)*Ni+i+1]*B[(j+1)*Nk+k+1];
			}
		}
	}
}

void aTbT_par01(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk) //(minn >= (Nk>>6)
{
	int i, j, k;
	int remi = Ni&1;
	if (remi) {
		for (k = 0; k < Nk; k++)
			for (j = 0; j < Nj; j++)
				C[j] += A[k*Ni]*B[j*Nk+k];
	}
	#pragma omp parallel for private(i, j)
	for (i=remi; i<Ni; i+=2) {
		int remk = Nk&1;
		if (remk) {
			for (j=0; j<Nj; j++) {
				C[i*Nj+j] += A[i]*B[j*Nk];
				C[(i+1)*Nj+j] += A[i+1]*B[j*Nk];
			}
		}

		for (k=remk; k<Nk; k+=2) {
			for (j=0; j<Nj; j++) {
				C[i*Nj+j] += A[k*Ni+i]*B[j*Nk+k];
				C[(i+1)*Nj+j] += A[k*Ni+i+1]*B[j*Nk+k];
				C[i*Nj+j] += A[(k+1)*Ni+i]*B[j*Nk+k+1];
				C[(i+1)*Nj+j] += A[(k+1)*Ni+i+1]*B[j*Nk+k+1];
			}
		}
	}
}

void aTbT_par10(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk) //// ikj notice, here i and j are relatively small, and k is very large
{
	int i, j, k;
	#pragma omp parallel for private(i, j, k)
	for (i = 0; i < Ni; i++) {
		int remk = Nk&7;
		for (k=0; k<remk; k++)
			for (j=0; j<Nj; j++)
				C[i*Nj+j] += A[k*Ni+i]*B[j*Nk+k];

		for (k=remk; k<Nk; k+=8) {
				for (j = 0; j < Nj; j++)
					C[i*Nj+j] = C[i*Nj+j] + A[k*Ni+i]*B[j*Nk+k] + A[(k+1)*Ni+i]*B[j*Nk+k+1] + A[(k+2)*Ni+i]*B[j*Nk+k+2] + A[(k+3)*Ni+i]*B[j*Nk+k+3] + A[(k+4)*Ni+i]*B[j*Nk+k+4] \
								+ A[(k+5)*Ni+i]*B[j*Nk+k+5] + A[(k+6)*Ni+i]*B[j*Nk+k+6] + A[(k+7)*Ni+i]*B[j*Nk+k+7];
		}
	}
}
