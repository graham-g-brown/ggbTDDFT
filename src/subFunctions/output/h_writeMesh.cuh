void h_writeMesh (
	meshStruct mesh
)
{
    FILE * IO;

    IO = fopen(LOCAL_OUTPUT_MESH_R, "wb");
    fwrite(mesh.h_r, sizeof(double), N_r, IO);
    fclose(IO);

	IO = fopen(LOCAL_OUTPUT_MESH_DRDX, "wb");
    fwrite(mesh.h_drdx, sizeof(double), N_r, IO);
    fclose(IO);

	IO = fopen(LOCAL_OUTPUT_MESH_TIME, "wb");
    fwrite(mesh.h_t, sizeof(double), N_t, IO);
    fclose(IO);
}

__global__
void d_getDiff (hartreeStruct hartree)
{
	int rdx = (blockIdx.x * blockDim.x) + threadIdx.x;

	hartree.d_v[rdx].x = hartree.d_v0[rdx].x;
}
