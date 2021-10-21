__global__
void d_calculateHartreePotential (
	hartreeStruct hartree,
	ksoSetTDStruct ksoSetTD,
	meshStruct mesh
)
{
	int radx = (blockIdx.x * blockDim.x) + threadIdx.x;
	int ldx  = (blockIdx.y * blockDim.y) + threadIdx.y;

	uint rbdx;

	double rmax, rmin;

	double2 v;

	v.x = 0.0;
	v.y = 0.0;

	if (ldx < N_har)
	{
		for (rbdx = 0; rbdx < N_r; rbdx ++)
		{
			if (radx >= rbdx)
			{
				rmax = mesh.d_r[radx];
				rmin = mesh.d_r[rbdx];
			}
			else
			{
				rmax = mesh.d_r[rbdx];
				rmin = mesh.d_r[radx];
			}

			if (ldx == 0)
			{
				v.x += hartree.d_density[rbdx].x / rmax;
			}
			else if (ldx == 1)
			{
				v.x += hartree.d_density[rbdx + ldx * N_r].x * rmin / pow(rmax, 2.0);
			}
			else if (ldx == 2)
			{
				v.x += hartree.d_density[rbdx + 2 * N_r].x * pow(rmin, 2.0) / pow(rmax, 3.0);
			}
			else if (ldx == 3)
			{
				v.x += hartree.d_density[rbdx + 3 * N_r].x * pow(rmin, 3.0) / pow(rmax, 4.0);
			}
		}
	}

	hartree.d_v[radx + ldx * N_r].x = v.x;
	hartree.d_v[radx + ldx * N_r].y = 0.0;
}

void h_calculateHartreePotential (
	hartreeStruct hartree,
	ksoSetTDStruct ksoSetTD,
	meshStruct mesh
)
{
	dim3 blocks (N_r / MTPB, N_har, 1);
	dim3 thread (MTPB, 1, 1);

	if ((USE_HARTREE == 1) && (targetID > 0))
	{
		d_getHartreePotentialDensity <<< blocks, thread >>> (hartree, ksoSetTD);
		cudaDeviceSynchronize();
		d_calculateHartreePotential <<< blocks, thread >>> (hartree, ksoSetTD, mesh);
	}
}







__global__
void d_calculateHartreePotential0 (
	hartreeStruct hartree,
	ksoSetTDStruct ksoSetTD,
	meshStruct mesh
)
{
	int radx = (blockIdx.x * blockDim.x) + threadIdx.x;
	int ldx  = (blockIdx.y * blockDim.y) + threadIdx.y;

	uint rbdx;

	double rmax;

	double2 v;

	v.x = 0.0;
	v.y = 0.0;

	for (rbdx = 0; rbdx < N_r; rbdx ++)
	{
		if (radx >= rbdx)
		{
			rmax = mesh.d_r[radx];
		}
		else
		{
			rmax = mesh.d_r[rbdx];
		}

		if (ldx == 0)
		{
			v.x += hartree.d_density0[rbdx].x / rmax;
		}
	}
	hartree.d_v0[radx].x =  v.x;
	hartree.d_v0[radx].y =  0.0;
}

void h_calculateHartreePotential0 (
	hartreeStruct hartree,
	ksoSetTDStruct ksoSetTD,
	meshStruct mesh
)
{
	dim3 blocks (N_r / MTPB, 1, 1);
	dim3 thread (MTPB, 1, 1);

	if ((USE_HARTREE == 1) && (targetID > 0))
	{
		d_getHartreePotentialDensity0 <<< blocks, thread >>> (hartree, ksoSetTD);
		cudaDeviceSynchronize();
		d_calculateHartreePotential0 <<< blocks, thread >>> (hartree, ksoSetTD, mesh);
		cudaDeviceSynchronize();
	}
}
