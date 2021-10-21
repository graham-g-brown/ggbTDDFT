__global__
void d_rotateLP1 (
	ksoSetTDStruct ksoSetTD,
	propagatorsStruct propagators
)
{
	int rdx   = (blockIdx.x * blockDim.x) + threadIdx.x;
	int ksodx = (blockIdx.y * blockDim.y) + threadIdx.y;
	int l0dx  = (blockIdx.z * blockDim.z) + threadIdx.z;

	int m = ksoSetTD.d_m[ksodx];
	int ldx;

	double2 temp;

	temp.x = 0.0;
	temp.y = 0.0;

	if (l0dx >= m)
	{
		for (ldx = m; ldx < N_l; ldx ++)
		{
			temp.x += propagators.d_LPV[m][l0dx + ldx * N_l] * ksoSetTD.d_x[ldx][rdx + ksodx * N_r].x;
			temp.y += propagators.d_LPV[m][l0dx + ldx * N_l] * ksoSetTD.d_x[ldx][rdx + ksodx * N_r].y;
		}
	}

	ksoSetTD.d_xROT[l0dx][rdx + ksodx * N_r] = temp;
}

__global__
void d_rotateLP2 (
	ksoSetTDStruct ksoSetTD,
	propagatorsStruct propagators
)
{
	int rdx   = (blockIdx.x * blockDim.x) + threadIdx.x;
	int ksodx = (blockIdx.y * blockDim.y) + threadIdx.y;
	int l0dx  = (blockIdx.z * blockDim.z) + threadIdx.z;

	int m = ksoSetTD.d_m[ksodx];
	int ldx;

	double2 temp;

	temp.x = 0.0;
	temp.y = 0.0;

	if (l0dx >= m)
	{
		for (ldx = m; ldx < N_l; ldx ++)
		{
			temp.x += propagators.d_LPV[m][ldx + l0dx * N_l] * ksoSetTD.d_xROT[ldx][rdx + ksodx * N_r].x;
			temp.y += propagators.d_LPV[m][ldx + l0dx * N_l] * ksoSetTD.d_xROT[ldx][rdx + ksodx * N_r].y;
		}
	}

	ksoSetTD.d_x[l0dx][rdx + ksodx * N_r] = temp;
}

__global__
void d_propagateLP (
	ksoSetTDStruct ksoSetTD,
	hartreeStruct hartree,
	propagatorsStruct propagators,
	meshStruct mesh,
	int tdx
)
{

	int rdx   = (blockIdx.x * blockDim.x) + threadIdx.x;
	int ksodx = (blockIdx.y * blockDim.y) + threadIdx.y;
	int ldx   = (blockIdx.z * blockDim.z) + threadIdx.z;

	double2 temp;

	double eta;

	int m = ksoSetTD.d_m[ksodx];

	if (ldx >= m)
	{
		if ((USE_HARTREE == 1) && (N_har > 1) && (targetID > 0))
		{
			if (ksodx >= 0)
			{
				eta = mesh.d_r[rdx] * mesh.d_E[tdx] + hartree.d_v[rdx + N_r].x;
			}
			else
			{
				eta = hartree.d_v[rdx + N_r].x;
			}
		}
		else
		{
			eta = mesh.d_r[rdx] * mesh.d_E[tdx];
		}

		eta *= (0.5 * dt) * propagators.d_LPD[m][ldx];

		temp.x = cos(eta) * ksoSetTD.d_xROT[ldx][rdx + ksodx * N_r].x + sin(eta) * ksoSetTD.d_xROT[ldx][rdx + ksodx * N_r].y;
		temp.y = cos(eta) * ksoSetTD.d_xROT[ldx][rdx + ksodx * N_r].y - sin(eta) * ksoSetTD.d_xROT[ldx][rdx + ksodx * N_r].x;

		ksoSetTD.d_xROT[ldx][rdx + ksodx * N_r] = temp;
	}
}

void h_couplingP (
	ksoSetTDStruct ksoSetTD,
	hartreeStruct hartree,
	meshStruct mesh,
	int tdx,
	int direction,
	propagatorsStruct propagators
)
{
	dim3 blocks (N_r / MTPB, N_kso, N_l);
	dim3 thread (MTPB, 1, 1);

	d_rotateLP1 <<< blocks, thread >>> (ksoSetTD, propagators);

	d_propagateLP <<< blocks, thread >>> (ksoSetTD, hartree, propagators, mesh, tdx);

	d_rotateLP2 <<< blocks, thread >>> (ksoSetTD, propagators);
}

__global__
void d_propagateLFP (
	ksoSetTDStruct ksoSetTD,
	hartreeStruct hartree,
	propagatorsStruct propagators,
	meshStruct mesh,
	int tdx
)
{

	int rdx   = (blockIdx.x * blockDim.x) + threadIdx.x;
	int ksodx = (blockIdx.y * blockDim.y) + threadIdx.y;
	int ldx   = (blockIdx.z * blockDim.z) + threadIdx.z;

	double2 temp;

	double eta;

	int m = ksoSetTD.d_m[ksodx];

	if (ldx >= m)
	{
		if ((USE_HARTREE == 1) && (N_har >= 3) && (targetID > 0))
		{
			eta = hartree.d_v[rdx + 3 * N_r].x;

			eta *= (0.5 * dt) * propagators.d_LFPD[m][ldx];

			temp.x = cos(eta) * ksoSetTD.d_xROT[ldx][rdx + ksodx * N_r].x + sin(eta) * ksoSetTD.d_xROT[ldx][rdx + ksodx * N_r].y;
			temp.y = cos(eta) * ksoSetTD.d_xROT[ldx][rdx + ksodx * N_r].y - sin(eta) * ksoSetTD.d_xROT[ldx][rdx + ksodx * N_r].x;

			ksoSetTD.d_xROT[ldx][rdx + ksodx * N_r] = temp;
		}
	}
}

__global__
void d_rotateLFP2 (
	ksoSetTDStruct ksoSetTD,
	propagatorsStruct propagators
)
{
	int rdx   = (blockIdx.x * blockDim.x) + threadIdx.x;
	int ksodx = (blockIdx.y * blockDim.y) + threadIdx.y;
	int l0dx  = (blockIdx.z * blockDim.z) + threadIdx.z;

	int m = ksoSetTD.d_m[ksodx];
	int ldx;

	double2 temp;

	temp.x = 0.0;
	temp.y = 0.0;

	if (l0dx >= m)
	{
		for (ldx = m; ldx < N_l; ldx ++)
		{
			temp.x += propagators.d_LFPV[m][ldx + l0dx * N_l] * ksoSetTD.d_xROT[ldx][rdx + ksodx * N_r].x;
			temp.y += propagators.d_LFPV[m][ldx + l0dx * N_l] * ksoSetTD.d_xROT[ldx][rdx + ksodx * N_r].y;
		}
	}

	ksoSetTD.d_x[l0dx][rdx + ksodx * N_r] = temp;
}

__global__
void d_rotateLFP1 (
	ksoSetTDStruct ksoSetTD,
	propagatorsStruct propagators
)
{
	int rdx   = (blockIdx.x * blockDim.x) + threadIdx.x;
	int ksodx = (blockIdx.y * blockDim.y) + threadIdx.y;
	int l0dx  = (blockIdx.z * blockDim.z) + threadIdx.z;

	int m = ksoSetTD.d_m[ksodx];
	int ldx;

	double2 temp;

	temp.x = 0.0;
	temp.y = 0.0;

	if (l0dx >= m)
	{
		for (ldx = m; ldx < N_l; ldx ++)
		{
			temp.x += propagators.d_LFPV[m][l0dx + ldx * N_l] * ksoSetTD.d_x[ldx][rdx + ksodx * N_r].x;
			temp.y += propagators.d_LFPV[m][l0dx + ldx * N_l] * ksoSetTD.d_x[ldx][rdx + ksodx * N_r].y;
		}
	}

	ksoSetTD.d_xROT[l0dx][rdx + ksodx * N_r] = temp;
}

void h_couplingFP (
	ksoSetTDStruct ksoSetTD,
	hartreeStruct hartree,
	meshStruct mesh,
	int tdx,
	int direction,
	propagatorsStruct propagators
)
{
	dim3 blocks (N_r / MTPB, N_kso, N_l);
	dim3 thread (MTPB, 1, 1);

	d_rotateLFP1 <<< blocks, thread >>> (ksoSetTD, propagators);

	d_propagateLFP <<< blocks, thread >>> (ksoSetTD, hartree, propagators, mesh, tdx);

	d_rotateLFP2 <<< blocks, thread >>> (ksoSetTD, propagators);
}
