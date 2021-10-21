__global__
void d_prepDipoleAcceleration (
	ksoSetTDStruct ksoSetTD,
	meshStruct mesh,
	tempStruct temp,
	int ldx,
	int ksodx
)
{
	int rdx = (blockIdx.x * blockDim.x) + threadIdx.x;

	double2 dip;

	dip.x = 0.0;
	dip.y = 0.0;

	double f_r = pow(mesh.d_r[rdx], - 2.0);

	if (ldx < ksoSetTD.d_m[ksodx])
	{
		dip.x = 0;
		dip.y = 0;
	}
	else if (ldx == ksoSetTD.d_m[ksodx])
	{
		dip.x = f_r *  d_cu (ldx, ksoSetTD.d_m[ksodx]) * ksoSetTD.d_x[ldx + 1][rdx + ksodx * N_r].x;
		dip.y = f_r *  d_cu (ldx, ksoSetTD.d_m[ksodx]) * ksoSetTD.d_x[ldx + 1][rdx + ksodx * N_r].y;
	}
	else if (ldx == N_l - 1)
	{
		dip.x = f_r *  d_cd (ldx, ksoSetTD.d_m[ksodx]) * ksoSetTD.d_x[ldx - 1][rdx + ksodx * N_r].x;
		dip.y = f_r *  d_cd (ldx, ksoSetTD.d_m[ksodx]) * ksoSetTD.d_x[ldx - 1][rdx + ksodx * N_r].y;
	}
	else
	{
		dip.x = f_r * (d_cu (ldx, ksoSetTD.d_m[ksodx]) * ksoSetTD.d_x[ldx + 1][rdx + ksodx * N_r].x
			  		 + d_cd (ldx, ksoSetTD.d_m[ksodx]) * ksoSetTD.d_x[ldx - 1][rdx + ksodx * N_r].x);

		dip.y = f_r * (d_cu (ldx, ksoSetTD.d_m[ksodx]) * ksoSetTD.d_x[ldx + 1][rdx + ksodx * N_r].y
					 + d_cd (ldx, ksoSetTD.d_m[ksodx]) * ksoSetTD.d_x[ldx - 1][rdx + ksodx * N_r].y);
	}

	temp.d_Z1[rdx] = dip;
}

void h_calculateDipoleAcceleration (
	ksoSetTDStruct ksoSetTD,
	meshStruct mesh,
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
			d_prepDipoleAcceleration <<< blocks, thread >>> (ksoSetTD, mesh, temp, ldx, ksodx);
			cudaDeviceSynchronize();

			cublasZdotc(handles.h_blas, N_r, ksoSetTD.h_d_x[ldx] + ksodx * N_r, 1, temp.d_Z1, 1, observables.h_dipoleAcceleration + tdx + ksodx * N_t + ldx * N_t * N_kso);
			cudaDeviceSynchronize();
		}
	}
}
