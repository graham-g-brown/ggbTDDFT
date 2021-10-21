__global__
void d_initialiseksoSetTD01 (
	ksoSetTDStruct ksoSetTD,
	int idx
)
{
	int rdx = (blockIdx.x * blockDim.x) + threadIdx.x;
    int ldx = (blockIdx.y * blockDim.y) + threadIdx.y;

	uint ksodx;

	for (ksodx = idx; ksodx < N_kso; ksodx ++)
	{
		ksoSetTD.d_x[ldx][rdx + ksodx * N_r].x = 0.0;
		ksoSetTD.d_x[ldx][rdx + ksodx * N_r].y = 0.0;
	}
}

void h_initialiseTDKSOSet (
	ksoSetTDStruct ksoSetTD,
	hartreeStruct hartree
)
{
	FILE * IO;

	int ksodx, rdx, ldx;

	char filenameSET[sizeof("./workingData/eigenStates/active/000/eigenStatesL000.bin")];

	for (ldx = 0; ldx < N_har; ldx ++)
	{
		for (rdx = 0; rdx < N_r; rdx ++)
		{
			hartree.h_v[rdx + ldx * N_r].x = 0.0;
			hartree.h_v[rdx + ldx * N_r].y = 0.0;

			hartree.h_density[rdx + ldx * N_r].x = 0.0;
			hartree.h_density[rdx + ldx * N_r].y = 0.0;
		}
	}

	cudaMemcpy (hartree.d_v, hartree.h_v, N_har * N_r * sizeof(double2), cudaMemcpyHostToDevice);
	cudaMemcpy (hartree.d_density, hartree.h_density, N_har * N_r * sizeof(double2), cudaMemcpyHostToDevice);

	for (ldx = 0; ldx < N_l; ldx ++)
	{
		for (ksodx = 0; ksodx < N_kso; ksodx ++)
		{
			for (rdx = 0; rdx < N_r; rdx ++)
			{
				ksoSetTD.h_x[ldx][rdx + ksodx * N_r].x = 0.0;
				ksoSetTD.h_x[ldx][rdx + ksodx * N_r].y = 0.0;
			}
		}
	}

	for (ldx = 0; ldx <= L0; ldx ++)
	{
		sprintf(filenameSET, "./workingData/eigenStates/active/%03d/eigenStatesL%03d.bin", targetID, ldx);

		IO = fopen(filenameSET, "rb");
	    fread(ksoSetTD.h_x[ldx], sizeof(double2), N_kso * N_r, IO);
	    fclose(IO);
	}

	IO = fopen("./workingData/stateParameters/l_active.bin", "rb");
	fread(ksoSetTD.h_l, sizeof(int), N_kso, IO);
	fclose(IO);
	cudaMemcpy (ksoSetTD.d_l, ksoSetTD.h_l, N_kso * sizeof(int), cudaMemcpyHostToDevice);

	for (ksodx = 0; ksodx < N_kso; ksodx ++)
	{
		for (ldx = 0; ldx < N_l; ldx ++)
		{
			if (ldx != ksoSetTD.h_l[ksodx])
			{
				for (rdx = 0; rdx < N_r; rdx ++)
				{
					ksoSetTD.h_x[ldx][rdx + ksodx * N_r].x = 0.0;
					ksoSetTD.h_x[ldx][rdx + ksodx * N_r].y = 0.0;
				}
			}
		}
	}

	for (ldx = 0; ldx < N_l; ldx ++)
	{
		cudaMemcpy (ksoSetTD.h_d_x[ldx], ksoSetTD.h_x[ldx], N_kso * N_r * sizeof(double2), cudaMemcpyHostToDevice);
	}

	IO = fopen("./workingData/stateParameters/m_active.bin", "rb");
	fread(ksoSetTD.h_m, sizeof(int), N_kso, IO);
	fclose(IO);
	cudaMemcpy (ksoSetTD.d_m, ksoSetTD.h_m, N_kso * sizeof(int), cudaMemcpyHostToDevice);

	IO = fopen("./workingData/stateParameters/occ_active.bin", "rb");
	fread(ksoSetTD.h_occ, sizeof(int), N_kso, IO);
	fclose(IO);

	cudaMemcpy (ksoSetTD.d_occ, ksoSetTD.h_occ, N_kso * sizeof(int), cudaMemcpyHostToDevice);

	IO = fopen("./workingData/stateParameters/densityFrozen.bin", "rb");
	fread(hartree.h_densityFrozen, sizeof(double), N_r, IO);
	fclose(IO);

	cudaMemcpy (hartree.d_densityFrozen, hartree.h_densityFrozen, N_r * sizeof(double), cudaMemcpyHostToDevice);
}

void h_writeTDKSOSet (
	ksoSetTDStruct ksoSetTD,
	int tdx
)
{
	FILE * IO;

	int ldx;

	char filenameSET[sizeof("../output/00000000/00000/timeDependent/TDKSO/L000/ksoL000_0000.bin")];

	for (ldx = 0; ldx < N_l; ldx ++)
	{
		sprintf(filenameSET, "../output/%08d/%05d/timeDependent/TDKSO/L%03d/ksoL%03d_%04d.bin", SIM_DATE, SIM_INDEX, ldx, ldx, tdx);

		cudaMemcpy (ksoSetTD.h_x[ldx], ksoSetTD.h_d_x[ldx], N_kso * N_r * sizeof(double2), cudaMemcpyDeviceToHost);

		IO = fopen(filenameSET, "wb");
	    fwrite(ksoSetTD.h_x[ldx], sizeof(double2), N_kso * N_r, IO);
	    fclose(IO);
	}
}
