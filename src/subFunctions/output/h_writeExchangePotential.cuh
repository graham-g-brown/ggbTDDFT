void h_writeExchangePotential (
	hartreeStruct hartree
)
{
    FILE * IO;

	cudaMemcpy (hartree.h_vexc0, hartree.d_vexc0, N_r * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy (hartree.h_vexc , hartree.d_vexc , N_r * sizeof(double), cudaMemcpyDeviceToHost);

    IO = fopen(LOCAL_OUTPUT_VEXC_VEXC0, "wb");
    fwrite(hartree.h_vexc0, sizeof(double), N_r, IO);
    fclose(IO);

	IO = fopen(LOCAL_OUTPUT_VEXC_VEXC, "wb");
    fwrite(hartree.h_vexc, sizeof(double), N_r, IO);
    fclose(IO);
}
