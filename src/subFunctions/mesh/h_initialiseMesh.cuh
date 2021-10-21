void h_initialiseMesh (
	meshStruct mesh
)
{
	FILE * IO;

	if ((IO = fopen(WORKING_DATA_MESH_R,"rb")) != NULL)
	{
		fread(mesh.h_r, sizeof(double), N_r, IO);
		fclose(IO);
	}
	else
	{
		printf("\n File not found at the following directory:");
		printf("\n ");
		printf(WORKING_DATA_MESH_R);
		printf("\n");
		exit(0);
	}

	if ((IO = fopen(WORKING_DATA_MESH_P,"rb")) != NULL)
	{
		fread(mesh.h_P, sizeof(double), N_r, IO);
		fclose(IO);
	}
	else
	{
		printf("\n File not found at the following directory:");
		printf("\n ");
		printf(WORKING_DATA_MESH_P);
		printf("\n");
		exit(0);
	}

	if ((IO = fopen(WORKING_DATA_MESH_DRDX,"rb")) != NULL)
	{
		fread(mesh.h_drdx, sizeof(double), N_r, IO);
		fclose(IO);
	}
	else
	{
		printf("\n File not found at the following directory:");
		printf("\n ");
		printf(WORKING_DATA_MESH_DRDX);
		printf("\n");
		exit(0);
	}

	if ((IO = fopen(WORKING_DATA_MESH_TIME,"rb")) != NULL)
	{
		fread(mesh.h_t, sizeof(double), N_t, IO);
		fclose(IO);
	}
	else
	{
		printf("\n File not found at the following directory:");
		printf("\n ");
		printf(WORKING_DATA_MESH_TIME);
		printf("\n");
		exit(0);
	}

	if ((IO = fopen(WORKING_DATA_MESH_TD,"rb")) != NULL)
	{
		fread(mesh.h_TD, sizeof(double), N_scan, IO);
		fclose(IO);
	}
	else
	{
		printf("\n File not found at the following directory:");
		printf("\n ");
		printf(WORKING_DATA_MESH_TD);
		printf("\n");
		exit(0);
	}

	cudaMemcpy (mesh.d_r   , mesh.h_r   , N_r * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy (mesh.d_P   , mesh.h_P   , N_r * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy (mesh.d_drdx, mesh.h_drdx, N_r * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy (mesh.d_t   , mesh.h_t   , N_t * sizeof(double), cudaMemcpyHostToDevice);
}
