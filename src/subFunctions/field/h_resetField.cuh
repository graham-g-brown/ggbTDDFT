void h_resetField (
	ksoSetTDStruct ksoSetTD,
	meshStruct mesh,
	observableStruct observables,
	int scandx
)
{
	uint tdx, ksodx, ldx, rdx;

	// ksoSet

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

	for (ldx = 0; ldx < N_kso; ldx ++)
	{
		cudaMemcpy (ksoSetTD.h_d_x[ldx], ksoSetTD.h_x[ldx], N_kso * N_r * sizeof(double2), cudaMemcpyHostToDevice);
	}

	for (tdx = 0; tdx < N_t; tdx ++)
	{
		mesh.h_E[tdx] = E0 * exp(- pow(2.0 * mesh.h_t[tdx] / tau0, 10.0)) * cos(omega0 * mesh.h_t[tdx]);

		for (ksodx = 0; ksodx < N_kso; ksodx ++)
		{
			for (ldx = 0; ldx < N_l; ldx ++)
			{
				observables.h_dipoleMoment[tdx + ksodx * N_t + ldx * N_t * N_kso].x = 0.0;
				observables.h_dipoleMoment[tdx + ksodx * N_t + ldx * N_t * N_kso].y = 0.0;
			}
		}

	}

	cudaMemcpy (mesh.d_E   , mesh.h_E   , N_t * sizeof(double), cudaMemcpyHostToDevice);
}
