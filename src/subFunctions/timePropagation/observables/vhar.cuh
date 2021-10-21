__global__
void d_prepVHAR (
	ksoSetTDStruct ksoSetTD,
	hartreeStruct hartree,
	tempStruct temp,
	int ldx,
	int ksodx
)
{
	int rdx = (blockIdx.x * blockDim.x) + threadIdx.x;

	double2 vhar;

	vhar.x = 0.0;
	vhar.y = 0.0;

	double f_r = hartree.d_v[rdx + 1 * N_r].x;

	if (ldx == 0)
	{
		vhar.x = f_r * d_c (ldx, ksoSetTD.d_m[ksodx]) * ksoSetTD.d_x[ldx + 1][rdx + ksodx * N_r].x;
		vhar.y = f_r * d_c (ldx, ksoSetTD.d_m[ksodx]) * ksoSetTD.d_x[ldx + 1][rdx + ksodx * N_r].y;
	}
	else if (ldx == N_l - 1)
	{
		vhar.x = f_r * d_c (ldx - 1, ksoSetTD.d_m[ksodx]) * ksoSetTD.d_x[ldx - 1][rdx + ksodx * N_r].x;
		vhar.y = f_r * d_c (ldx - 1, ksoSetTD.d_m[ksodx]) * ksoSetTD.d_x[ldx - 1][rdx + ksodx * N_r].y;
	}
	else
	{
		vhar.x = f_r * (d_c (ldx, ksoSetTD.d_m[ksodx]) * ksoSetTD.d_x[ldx + 1][rdx + ksodx * N_r].x
			  		  + d_c (ldx - 1, ksoSetTD.d_m[ksodx]) * ksoSetTD.d_x[ldx - 1][rdx + ksodx * N_r].x);
		vhar.y = f_r * (d_c (ldx, ksoSetTD.d_m[ksodx]) * ksoSetTD.d_x[ldx + 1][rdx + ksodx * N_r].y
					  + d_c (ldx - 1, ksoSetTD.d_m[ksodx]) * ksoSetTD.d_x[ldx - 1][rdx + ksodx * N_r].y);
	}

	temp.d_Z1[rdx] = vhar;
}

void h_calculateVHAR (
	ksoSetTDStruct ksoSetTD,
	hartreeStruct hartree,
	tempStruct temp,
	observableStruct observables,
	handleStruct handles,
	int tdx
)
{
	dim3 blocks (N_r / MTPB, 1, 1);
	dim3 thread (MTPB, 1, 1);

	uint ksodx, ldx;

	for (ksodx = 0; ksodx < N_kso; ksodx ++)
	{
		for (ldx = 0; ldx < N_l; ldx ++)
		{
			d_prepVHAR <<< blocks, thread >>> (ksoSetTD, hartree, temp, ldx, ksodx);
			cudaDeviceSynchronize();

			cublasZdotc(handles.h_blas, N_r, ksoSetTD.h_d_x[ldx] + ksodx * N_r, 1, temp.d_Z1, 1, observables.h_vhar + tdx + ksodx * N_t + ldx * N_t * N_kso);
			cudaDeviceSynchronize();
		}
	}
}

__global__
void d_saveHartreeDipole (
	hartreeStruct hartree,
	observableStruct observables,
	int tdx
)
{
	int rdx = (blockIdx.x * blockDim.x) + threadIdx.x;

	observables.d_vharDipole[rdx + tdx * N_r] = hartree.d_v[rdx + N_r];
}

void h_saveHartreeDipole (
	hartreeStruct hartree,
	observableStruct observables,
	int tdx
)
{
	dim3 blocks (N_r / MTPB, 1, 1);
	dim3 thread (MTPB, 1, 1);

	d_saveHartreeDipole <<< blocks, thread >>> (hartree, observables, tdx);
}
