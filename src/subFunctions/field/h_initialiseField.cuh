void h_initialiseField (
	meshStruct mesh,
	int scandx,
	int end
)
{
	uint tdx;

	double t, tp, E, Ep;

	if (IN_SITU == 1)
	{
		for (tdx = 0; tdx < N_t; tdx ++)
		{
			if (end == 0)
			{
				t  = mesh.h_t[tdx];
				tp = mesh.h_t[tdx] * COS_THETA;

				E  = E0 * exp(- pow(2.0 * t / tau0, 10.0)) * cos(omega0 * t + CEP0);
				Ep = PERTURBATION_AMPLITUDE * E0 * exp(- pow(2.0 * tp / tau0, 10.0)) * cos(omega0 * (t + mesh.h_TD[scandx]) + CEP0);

				mesh.h_E[tdx] = E + Ep;
			}
			else
			{
				mesh.h_E[tdx] = E0 * exp(- pow(2.0 * mesh.h_t[tdx] / tau0, 10.0)) * cos(omega0 * mesh.h_t[tdx] + CEP0);
			}
		}
	}
	else
	{
		for (tdx = 0; tdx < N_t; tdx ++)
		{
			mesh.h_E[tdx] = E0 * exp(- pow(2.0 * mesh.h_t[tdx] / tau0, 10.0)) * cos(omega0 * mesh.h_t[tdx] + CEP0);
		}
	}

	cudaMemcpy (mesh.d_E, mesh.h_E, N_t * sizeof(double), cudaMemcpyHostToDevice);
}
