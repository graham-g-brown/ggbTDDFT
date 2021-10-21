void h_VRAMPrompt ()
{
    size_t memFree, total;
    cudaMemGetInfo(& memFree,& total);
    printf("     - VRAM Used          : %05lu MB / %05lu MB \n", (total - memFree)/1024/1024, total/1024/1024);
    printf("\n");
}

void h_updateTimePropagation ()
{
    printf("  4. Time Propagation \n\n");
	printf("     Scan\n\n");
}

void h_updateTimePropagationInSitu (int scandx)
{
    printf("  4.%02d.%02d Time Propagation \n\n", scandx, N_scan);
}

void h_updateTimePropagation2 ()
{
    printf("  4. No Time Propagation \n\n");
}

void h_checkCUDADevice (

)
{
    int devID = CUDA_DEVICE;
    int deviceCount;
    cudaDeviceProp deviceProp;
	int cudaVersion;

    cudaGetDeviceCount(&deviceCount);
    cudaGetDeviceProperties(&deviceProp, devID);

	size_t memFree, total;

    if (!cudaSetDevice(devID))
    {
		cudaMemGetInfo(& memFree,& total);
		cudaDriverGetVersion ( & cudaVersion);

		printf("\n");
		printf("  3. Computation Platform Information \n\n");
		printf("     - CUDA Version       : %d.%d \n", cudaVersion / 1000, (cudaVersion % 100) / 10);
		printf("     - Platform           : NVIDIA-CUDA\n");
        printf("     - Device %d (arch %d%d) : %s\n",
               devID, deviceProp.major, deviceProp.minor,  deviceProp.name);
		printf("     - VRAM               : %05lu MB \n", deviceProp.totalGlobalMem/1024/1024);
		printf("     - Multiprocessors    : %d \n", deviceProp.multiProcessorCount);
		printf("     - Shared RAM / Block : %lu \n",deviceProp.sharedMemPerBlock);
		printf("     - Registers / Block  : %d \n", deviceProp.regsPerBlock);
		printf("     - Warp size          : %d\n", deviceProp.warpSize);
    }

    else
    {
		printf("\n");
        printf("  Failure setting CUDA Device\n");
        printf("\n");
        exit(0);
    }
}

void h_printTDResults (

)
{
	printf("\n\n");
	printf("     - Time propagation finished without error");
	printf("\n");
}

static const char *_cudaGetErrorEnum(cublasStatus_t error)
{
    switch (error)
    {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS\n";

        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED\n";

        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED\n";

        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE\n";

        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH\n";

        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR\n";

        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED\n";

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR\n";
    }

    return "<unknown>\n";
}

void h_timePropagationProgressBar (
	double progress,
	int tdx,
	int scandx
)
{
	int barWidth = 100;
	std::cout << "     " << std::setfill('0') << std::setw(4) << scandx << "|" << std::setw(5) << tdx << "/" << N_t << ": │";
	// std::cout << "     " << scandx << " ";
	int pos = barWidth * progress;
	for (int i = 0; i < barWidth; ++i) {
		if (i < pos) std::cout << "█";
		else if (i == pos) std::cout << "█";
		else std::cout << "-";
	}
	std::cout << "│  " << int(progress * 100.0) << " %\r";
	std::cout.flush();
}
