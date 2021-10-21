void h_writeHartreePotential (
	hartreeStruct hartree
)
{
    FILE * IO;

	cudaMemcpy (hartree.h_v0, hartree.d_v0, N_r * sizeof(double2), cudaMemcpyDeviceToHost);
	cudaMemcpy (hartree.h_v , hartree.d_v , N_har * N_r * sizeof(double2), cudaMemcpyDeviceToHost);

    IO = fopen(LOCAL_OUTPUT_VHAR_VHAR0, "wb");
    fwrite(hartree.h_v0, sizeof(double2), N_r, IO);
    fclose(IO);

	IO = fopen(LOCAL_OUTPUT_VHAR_VHAR, "wb");
    fwrite(hartree.h_v, sizeof(double2), N_har * N_r, IO);
    fclose(IO);
}
