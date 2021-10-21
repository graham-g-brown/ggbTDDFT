__global__
void d_couplingS (
	ksoSetTDStruct ksoSetTD,
	hartreeStruct hartree,
	meshStruct mesh
)
{
	int rdx   = (blockIdx.x * blockDim.x) + threadIdx.x;

	int ldx   = (blockIdx.y * blockDim.y) + threadIdx.y;

	int ksodx = (blockIdx.z * blockDim.z) + threadIdx.z;

	double2 temp;

	double eta;

	if ((USE_HARTREE == 1) && (targetID > 0))
	{
		eta = hartree.d_v[rdx].x  - hartree.d_v0[rdx].x
			+ hartree.d_vexc[rdx] - hartree.d_vexc0[rdx];

		if (N_har > 2)
		{
			eta += d_p(ldx, ksoSetTD.d_m[ksodx]) * hartree.d_v[rdx + 2 * N_r].x;
		}

		eta *= 0.5 * dt;

		if ((ksodx < N_kso) && (ldx < N_l))
		{
			temp.x = cos(eta) * ksoSetTD.d_x[ldx][rdx + ksodx * N_r].x
				   + sin(eta) * ksoSetTD.d_x[ldx][rdx + ksodx * N_r].y;

			temp.y = cos(eta) * ksoSetTD.d_x[ldx][rdx + ksodx * N_r].y
				   - sin(eta) * ksoSetTD.d_x[ldx][rdx + ksodx * N_r].x;

			ksoSetTD.d_x[ldx][rdx + ksodx * N_r] = temp;
		}
	}
}

void h_couplingS (
	ksoSetTDStruct ksoSetTD,
	hartreeStruct hartree,
	meshStruct mesh
)
{
	if (N_har >= 3)
	{
		dim3 blocks (N_r / MTPB, N_l / MTPB, N_kso);
		dim3 thread (MTPB, MTPB, 1);

		d_couplingS <<< blocks, thread >>> (ksoSetTD, hartree, mesh);
		cudaDeviceSynchronize();
	}
}
