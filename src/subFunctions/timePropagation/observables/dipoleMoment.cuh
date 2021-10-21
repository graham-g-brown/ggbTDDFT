__global__
void d_prepDipoleMoment (
	ksoSetTDStruct ksoSetTD,
	meshStruct mesh,
	tempStruct temp,
	int ldx,
	int ksodx,
	int scandx
)
{
	int rdx = (blockIdx.x * blockDim.x) + threadIdx.x;

	double2 dip;
	double f_r;

	dip.x = 0.0;
	dip.y = 0.0;

	f_r = mesh.d_r[rdx];

	if (ldx < ksoSetTD.d_m[ksodx])
	{
		dip.x = 0;
		dip.y = 0;
	}
	else if (ldx == ksoSetTD.d_m[ksodx])
	{
		dip.x = f_r * d_cu (ldx, ksoSetTD.d_m[ksodx]) * ksoSetTD.d_x[ldx + 1][rdx + ksodx * N_r].x;
		dip.y = f_r * d_cu (ldx, ksoSetTD.d_m[ksodx]) * ksoSetTD.d_x[ldx + 1][rdx + ksodx * N_r].y;
	}
	else if (ldx == N_l - 1)
	{
		dip.x = f_r * d_cd (ldx, ksoSetTD.d_m[ksodx]) * ksoSetTD.d_x[ldx - 1][rdx + ksodx * N_r].x;
		dip.y = f_r * d_cd (ldx, ksoSetTD.d_m[ksodx]) * ksoSetTD.d_x[ldx - 1][rdx + ksodx * N_r].y;
	}
	else
	{
		dip.x = f_r * d_cu (ldx, ksoSetTD.d_m[ksodx]) * ksoSetTD.d_x[ldx + 1][rdx + ksodx * N_r].x
			  	+ f_r * d_cd (ldx, ksoSetTD.d_m[ksodx]) * ksoSetTD.d_x[ldx - 1][rdx + ksodx * N_r].x;

		dip.y = f_r * d_cu (ldx, ksoSetTD.d_m[ksodx]) * ksoSetTD.d_x[ldx + 1][rdx + ksodx * N_r].y
					+ f_r * d_cd (ldx, ksoSetTD.d_m[ksodx]) * ksoSetTD.d_x[ldx - 1][rdx + ksodx * N_r].y;
	}

	temp.d_Z1[rdx].x = dip.x;
	temp.d_Z1[rdx].y = dip.y;
}

void h_calculateDipoleMoment (
	ksoSetTDStruct ksoSetTD,
	meshStruct mesh,
	tempStruct temp,
	observableStruct observables,
	handleStruct handles,
	int tdx,
	int scandx
)
{
	dim3 blocks (N_r / MTPB, 1, 1);
	dim3 thread (MTPB, 1, 1);

	uint ksodx, ldx;

	for (ksodx = 0; ksodx < N_kso; ksodx ++)
	{
		for (ldx = 0; ldx < N_l; ldx ++)
		{
			d_prepDipoleMoment <<< blocks, thread >>> (ksoSetTD, mesh, temp, ldx, ksodx, scandx);
			cudaDeviceSynchronize();

			cublasZdotc(handles.h_blas, N_r, ksoSetTD.h_d_x[ldx] + ksodx * N_r, 1, temp.d_Z1, 1, observables.h_dipoleMoment + tdx + ksodx * N_t + ldx * N_t * N_kso);
			cudaDeviceSynchronize();
		}
	}
}
