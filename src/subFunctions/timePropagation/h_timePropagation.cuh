void h_propagationH0 (
	ksoSetTDStruct ksoSetTD,
	propagatorsStruct propagators,
	handleStruct handles
)
{
	cublasStatus_t STATUS;

	double2 alpha, beta;

	alpha.x = 1.0;
	alpha.y = 0.0;

	beta.x  = 0.0;
	beta.y  = 0.0;

	STATUS = cublasZgemmBatched(handles.h_blas,
					   			CUBLAS_OP_N,
					   			CUBLAS_OP_N,
					   			N_r, N_kso, N_r,
					   			& alpha,
					   			(const double2 **) propagators.d_W, N_r,
					   			(const double2 **) ksoSetTD.d_x   , N_r,
					   			& beta,
					   			ksoSetTD.d_x, N_r,
					   			N_l);

	assert(CUBLAS_STATUS_SUCCESS == STATUS);
	cudaDeviceSynchronize();
}

void h_timePropagation (
	ksoSetTDStruct ksoSetTD,
	hartreeStruct hartree,
	propagatorsStruct propagators,
	meshStruct mesh,
	observableStruct observables,
	tempStruct temp,
	handleStruct handles,
	int scandx
)
{
	int tdx;

	double progress = 0.0;

	h_calculateHartreePotential0 (hartree, ksoSetTD, mesh);
	cudaDeviceSynchronize();

	h_exchangePotential0 (hartree, mesh);
	cudaDeviceSynchronize();

	h_calculateHartreePotential (hartree, ksoSetTD, mesh);
	cudaDeviceSynchronize();

	h_exchangePotential (hartree, mesh);
	cudaDeviceSynchronize();

	for (tdx = 0; tdx < N_t; tdx ++)
	{

		h_couplingFF (ksoSetTD, hartree, mesh, tdx, 0, propagators);
		cudaDeviceSynchronize();

		h_couplingFP (ksoSetTD, hartree, mesh, tdx, 0, propagators);
		cudaDeviceSynchronize();

		h_couplingD (ksoSetTD, hartree, mesh, tdx, 0, propagators);
		cudaDeviceSynchronize();

		h_couplingP (ksoSetTD, hartree, mesh, tdx, 0, propagators);
		cudaDeviceSynchronize();

		h_couplingS (ksoSetTD, hartree, mesh);
		cudaDeviceSynchronize();

		h_propagationH0 (ksoSetTD, propagators, handles);
		cudaDeviceSynchronize();

		h_couplingS (ksoSetTD, hartree, mesh);
		cudaDeviceSynchronize();

		h_couplingP (ksoSetTD, hartree, mesh, tdx, 1, propagators);
		cudaDeviceSynchronize();

		h_couplingD (ksoSetTD, hartree, mesh, tdx, 1, propagators);
		cudaDeviceSynchronize();

		h_couplingFP (ksoSetTD, hartree, mesh, tdx, 1, propagators);
		cudaDeviceSynchronize();

		h_couplingFF (ksoSetTD, hartree, mesh, tdx, 1, propagators);
		cudaDeviceSynchronize();

		//

		h_calculateHartreePotential (hartree, ksoSetTD, mesh);
		cudaDeviceSynchronize();

		h_exchangePotential (hartree, mesh);
		cudaDeviceSynchronize();

		h_calculateDipoleMoment (ksoSetTD, mesh, temp, observables, handles, tdx, scandx);
		cudaDeviceSynchronize();

		h_calculateDipoleAcceleration (ksoSetTD, mesh, temp, observables, handles, tdx);
		cudaDeviceSynchronize();

		h_timePropagationProgressBar (progress, tdx, scandx);

		progress = double(tdx + 2) / double(N_t);
	}
	printf("\n");
}
