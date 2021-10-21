void h_writeDipoleMoment (
	observableStruct observables,
	int scandx
)
{
    FILE * IO;

	uint ksodx, ldx;

	char filename[sizeof("../output/20200000/00000/timeDependent/observables/dipoleMoment/scan000/kso000/dipL000.bin")];

	for (ksodx = 0; ksodx < N_kso; ksodx ++)
	{
		for (ldx = 0; ldx < N_l; ldx ++)
		{
			sprintf(filename, "../output/%08d/%05d/timeDependent/observables/dipoleMoment/scan%03d/kso%03d/dipL%03d.bin", SIM_DATE, SIM_INDEX, scandx, ksodx, ldx);

			IO = fopen(filename, "wb");
		    fwrite(observables.h_dipoleMoment + ksodx * N_t + ldx * N_t * N_kso, sizeof(double2), N_t, IO);
		    fclose(IO);
		}
	}
}

void h_writeDipoleAcceleration (
	observableStruct observables,
	int scandx
)
{
    FILE * IO;

	uint ksodx, ldx;

	char filename[sizeof("../output/20200000/00000/timeDependent/observables/dipoleAcceleration/scan000/kso000/dipL000.bin")];

	for (ksodx = 0; ksodx < N_kso; ksodx ++)
	{
		for (ldx = 0; ldx < N_l; ldx ++)
		{
			sprintf(filename, "../output/%08d/%05d/timeDependent/observables/dipoleAcceleration/scan%03d/kso%03d/dipL%03d.bin", SIM_DATE, SIM_INDEX, scandx, ksodx, ldx);

			IO = fopen(filename, "wb");
		    fwrite(observables.h_dipoleAcceleration + ksodx * N_t + ldx * N_t * N_kso, sizeof(double2), N_t, IO);
		    fclose(IO);
		}
	}
}

void h_writeVHAR (
	observableStruct observables,
	int scandx
)
{
    FILE * IO;

	uint ksodx, ldx;

	char filename[sizeof("../output/20200000/00000/timeDependent/observables/vhar/scan000/kso000/vharL000.bin")];

	for (ksodx = 0; ksodx < N_kso; ksodx ++)
	{
		for (ldx = 0; ldx < N_l; ldx ++)
		{
			sprintf(filename, "../output/%08d/%05d/timeDependent/observables/vhar/scan%03d/kso%03d/vharL%03d.bin", SIM_DATE, SIM_INDEX, scandx, ksodx, ldx);

			IO = fopen(filename, "wb");
		    fwrite(observables.h_vhar + ksodx * N_t + ldx * N_t * N_kso, sizeof(double2), N_t, IO);
		    fclose(IO);
		}
	}
}

void h_writeField (
	meshStruct mesh,
	int scandx
)
{
    FILE * IO;

	char filename[sizeof("../output/20200000/00000/timeDependent/field/E000.bin")];

	sprintf(filename, "../output/%08d/%05d/timeDependent/field/E%03d.bin", SIM_DATE, SIM_INDEX, scandx);

	IO = fopen(filename, "wb");
    fwrite(mesh.h_E, sizeof(double), N_t, IO);
    fclose(IO);
}

void h_writeDipoleMomentInSitu (
	observableStruct observables,
	int scandx
)
{
    FILE * IO;

	uint ksodx, ldx;

	char filename[sizeof("../output/20200000/00000/timeDependent/observables/dipoleMoment/%02d/kso000/dipL000.bin")];

	for (ksodx = 0; ksodx < N_kso; ksodx ++)
	{
		for (ldx = 0; ldx < N_l; ldx ++)
		{
			sprintf(filename, "../output/%08d/%05d/timeDependent/observables/dipoleMoment/%02d/kso%03d/dipL%03d.bin", SIM_DATE, SIM_INDEX, scandx, ksodx, ldx);

			IO = fopen(filename, "wb");
		    fwrite(observables.h_dipoleMoment + ksodx * N_t + ldx * N_t * N_kso, sizeof(double2), N_t, IO);
		    fclose(IO);
		}
	}
}

void h_writeTDWF (
	ksoSetTDStruct ksoSetTD
)
{
    FILE * IO;

	uint ldx;

	char filename[sizeof("../output/20200000/00000/timeDependent/kso000.bin")];

	ldx = 0;
	cudaMemcpy (ksoSetTD.h_x[ldx], ksoSetTD.h_d_x[ldx], N_r * N_kso * sizeof(double2), cudaMemcpyDeviceToHost);

	sprintf(filename, "../output/%08d/%05d/timeDependent/kso%03d.bin", SIM_DATE, SIM_INDEX, ldx);

	IO = fopen(filename, "wb");
    fwrite(ksoSetTD.h_x[ldx], sizeof(double2), N_r, IO);
    fclose(IO);

	ldx = 1;
	cudaMemcpy (ksoSetTD.h_x[ldx], ksoSetTD.h_d_x[ldx], N_r * N_kso * sizeof(double2), cudaMemcpyDeviceToHost);
	sprintf(filename, "../output/%08d/%05d/timeDependent/kso%03d.bin", SIM_DATE, SIM_INDEX, ldx);
	IO = fopen(filename, "wb");
    fwrite(ksoSetTD.h_x[ldx], sizeof(double2), N_r, IO);
    fclose(IO);
}
