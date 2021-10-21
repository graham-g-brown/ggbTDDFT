__global__
void d_exchangePotential0 (
	hartreeStruct hartree,
	meshStruct mesh
)
{
	int rdx = (blockIdx.x * blockDim.x) + threadIdx.x;

	double density0 = N_g * (N_g + 1) / 2.0 * (hartree.d_density0[rdx].x / 4.0 / M_PI + hartree.d_densityFrozen[rdx]) / mesh.d_drdx[rdx] * pow(mesh.d_P[rdx], 2.0);

	hartree.d_vexc0[rdx]  = - pow(3.0 / M_PI * density0, 1.0 / 3.0);
	hartree.d_vexc0[rdx] /= pow(mesh.d_r[rdx], 2.0 / 3.0);
}

void h_exchangePotential0 (
	hartreeStruct hartree,
	meshStruct mesh
)
{
	dim3 blocks (N_r / MTPB, 1, 1);
	dim3 thread (MTPB, 1, 1);

	d_exchangePotential0 <<< blocks, thread >>> (hartree, mesh);
}

__global__
void d_exchangePotential (
	hartreeStruct hartree,
	meshStruct mesh
)
{
	int rdx = (blockIdx.x * blockDim.x) + threadIdx.x;

	double density0 = N_g * (N_g + 1) / 2.0 * (hartree.d_density[rdx].x / 4.0 / M_PI + hartree.d_densityFrozen[rdx]) / mesh.d_drdx[rdx] * pow(mesh.d_P[rdx], 2.0);

	double den = density0;

	if (den < 0.0)
	{
		den = 0.0;
	}

	hartree.d_vexc[rdx]  =  - pow(3.0 / M_PI * den, 1.0 / 3.0);
	hartree.d_vexc[rdx] /= pow(mesh.d_r[rdx], 2.0 / 3.0);
}

void h_exchangePotential (
	hartreeStruct hartree,
	meshStruct mesh
)
{
	dim3 blocks (N_r / MTPB, 1, 1);
	dim3 thread (MTPB, 1, 1);

	d_exchangePotential <<< blocks, thread >>> (hartree, mesh);
}
