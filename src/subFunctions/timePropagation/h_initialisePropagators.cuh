void h_initialisePropagators (
	propagatorsStruct propagators
)
{
	FILE * IO;

	int l;

	// Monopole

	char filenameL[sizeof("./workingData/hamiltonian/propagators/propagatorL000.bin")];

	for (l = 0; l < N_l; l ++)
	{
		sprintf(filenameL, "./workingData/hamiltonian/propagators/propagatorL%03d.bin", l);

		IO = fopen(filenameL,"rb");
	    fread(propagators.h_W[l], sizeof(double2), N_r * N_r, IO);
	    fclose(IO);

		cudaMemcpy (propagators.h_d_W[l], propagators.h_W[l], N_r * N_r * sizeof(double2), cudaMemcpyHostToDevice);
	}

	// Dipole

	char filenamePV[sizeof("./workingData/hamiltonian/propagatorsL/001/001/V000.bin")];
	char filenamePD[sizeof("./workingData/hamiltonian/propagatorsL/001/001/D000.bin")];

	for (l = 0; l < N_m; l ++)
	{
		sprintf(filenamePV, "./workingData/hamiltonian/propagatorsL/001/001/V%03d.bin", l);

		IO = fopen(filenamePV,"rb");
	    fread(propagators.h_LPV[l], sizeof(double), N_l * N_l, IO);
	    fclose(IO);

		cudaMemcpy (propagators.h_d_LPV[l], propagators.h_LPV[l], N_l * N_l * sizeof(double), cudaMemcpyHostToDevice);

		sprintf(filenamePD, "./workingData/hamiltonian/propagatorsL/001/001/D%03d.bin", l);

		IO = fopen(filenamePD,"rb");
	    fread(propagators.h_LPD[l], sizeof(double), N_l, IO);
	    fclose(IO);

		cudaMemcpy (propagators.h_d_LPD[l], propagators.h_LPD[l], N_l * sizeof(double), cudaMemcpyHostToDevice);
	}

	// Quadrupole

	char filenameDV[sizeof("./workingData/hamiltonian/propagatorsL/002/002/V000.bin")];
	char filenameDD[sizeof("./workingData/hamiltonian/propagatorsL/002/002/D000.bin")];

	for (l = 0; l < N_m; l ++)
	{
		sprintf(filenameDV, "./workingData/hamiltonian/propagatorsL/002/002/V%03d.bin", l);

		IO = fopen(filenameDV,"rb");
	    fread(propagators.h_LDV[l], sizeof(double), N_l * N_l, IO);
	    fclose(IO);

		cudaMemcpy (propagators.h_d_LDV[l], propagators.h_LDV[l], N_l * N_l * sizeof(double), cudaMemcpyHostToDevice);

		sprintf(filenameDD, "./workingData/hamiltonian/propagatorsL/002/002/D%03d.bin", l);

		IO = fopen(filenameDD,"rb");
	    fread(propagators.h_LDD[l], sizeof(double), N_l, IO);
	    fclose(IO);

		cudaMemcpy (propagators.h_d_LDD[l], propagators.h_LDD[l], N_l * sizeof(double), cudaMemcpyHostToDevice);
	}

	// Octupole - P

	char filenameFPV[sizeof("./workingData/hamiltonian/propagatorsL/001/003/V000.bin")];
	char filenameFPD[sizeof("./workingData/hamiltonian/propagatorsL/001/003/D000.bin")];

	for (l = 0; l < N_m; l ++)
	{
		sprintf(filenameFPV, "./workingData/hamiltonian/propagatorsL/001/003/V%03d.bin", l);

		IO = fopen(filenameFPV,"rb");
	    fread(propagators.h_LFPV[l], sizeof(double), N_l * N_l, IO);
	    fclose(IO);

		cudaMemcpy (propagators.h_d_LFPV[l], propagators.h_LFPV[l], N_l * N_l * sizeof(double), cudaMemcpyHostToDevice);

		sprintf(filenameFPD, "./workingData/hamiltonian/propagatorsL/001/003/D%03d.bin", l);

		IO = fopen(filenameFPD,"rb");
	    fread(propagators.h_LFPD[l], sizeof(double), N_l, IO);
	    fclose(IO);

		cudaMemcpy (propagators.h_d_LFPD[l], propagators.h_LFPD[l], N_l * sizeof(double), cudaMemcpyHostToDevice);
	}

	// Octupole - P

	char filenameFFV[sizeof("./workingData/hamiltonian/propagatorsL/003/003/V000.bin")];
	char filenameFFD[sizeof("./workingData/hamiltonian/propagatorsL/003/003/D000.bin")];

	for (l = 0; l < N_m; l ++)
	{
		sprintf(filenameFFV, "./workingData/hamiltonian/propagatorsL/003/003/V%03d.bin", l);

		IO = fopen(filenameFFV,"rb");
	    fread(propagators.h_LFFV[l], sizeof(double), N_l * N_l, IO);
	    fclose(IO);

		cudaMemcpy (propagators.h_d_LFFV[l], propagators.h_LFFV[l], N_l * N_l * sizeof(double), cudaMemcpyHostToDevice);

		sprintf(filenameFFD, "./workingData/hamiltonian/propagatorsL/003/003/D%03d.bin", l);

		IO = fopen(filenameFFD,"rb");
	    fread(propagators.h_LFFD[l], sizeof(double), N_l, IO);
	    fclose(IO);

		cudaMemcpy (propagators.h_d_LFFD[l], propagators.h_LFFD[l], N_l * sizeof(double), cudaMemcpyHostToDevice);
	}
}
