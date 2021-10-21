// C Libraries

#include <stdio.h>
#include <stdlib.h>
#include <locale.h>
#include <wchar.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <assert.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <math.h>
#include <gsl/gsl_sf_laguerre.h>
#include <gsl/gsl_sf_legendre.h>
#include <gsl/gsl_sf_coulomb.h>
#include <gsl/gsl_sf_fermi_dirac.h>

#include <signal.h>

// CUDA Libraries

#include <cuda_runtime.h>
#include <cufft.h>
#include <cusolverDn.h>
#include <cuComplex.h>
#include <ctime>
#include <cstdio>
#include <complex>
#include <cublas_v2.h>

//Type Definition for Complex-Type
using namespace std;
typedef complex <double> long_double_complex;
long_double_complex Il = long_double_complex(0.0,1.0);

typedef struct
{
	// Host Pointers

	double * h_r;
	double * h_P;
	double * h_drdx;
	double * h_t;
	double * h_TD;
	double * h_E;

	double2 * h_D1;

	// Device Pointers

	double * d_r;
	double * d_P;
	double * d_drdx;
	double * d_t;
	double * d_E;

	double2 * d_D1;

} meshStruct;

typedef struct
{
	// Host Pointers

	double2 ** h_W;
	double2 ** h_d_W;

	// Dipole

	double  ** h_LPV;
	double  ** h_d_LPV;

	double  ** h_LPD;
	double  ** h_d_LPD;

	// Quadrupole

	double  ** h_LDV;
	double  ** h_d_LDV;

	double  ** h_LDD;
	double  ** h_d_LDD;

	// Octupole - P

	double  ** h_LFPV;
	double  ** h_d_LFPV;

	double  ** h_LFPD;
	double  ** h_d_LFPD;

	// Octupole - F

	double  ** h_LFFV;
	double  ** h_d_LFFV;

	double  ** h_LFFD;
	double  ** h_d_LFFD;

	// Device Pointers

	double2 ** d_W;

	// Dipole

	double  ** d_LPV;
	double  ** d_LPD;

	// Quadrupole

	double  ** d_LDV;
	double  ** d_LDD;

	// Octupole - P

	double  ** d_LFPV;
	double  ** d_LFPD;

	// Octupole - F

	double  ** d_LFFV;
	double  ** d_LFFD;

} propagatorsStruct;

typedef struct
{
	double ** h_propL;
	double ** h_d_propL;

	double ** d_propL;
} propagatorsLStruct;

typedef struct
{
	double2 ** h_x;
	double2 ** h_d_x;

	double2 ** h_xROT;
	double2 ** h_d_xROT;

	double2 * h_density;

	int * h_l;
	int * h_m;
	int * h_occ;

	int * d_l;
	int * d_m;
	int * d_occ;

	double2 ** d_x;
	double2 ** d_xROT;

	double2 * d_density;

} ksoSetTDStruct;

typedef struct
{
	double2 * h_dipoleMoment;
	double2 * h_dipoleAcceleration;
	double2 * h_vhar;

	double2 * h_vharDipole;

	double2 * d_vharDipole;

} observableStruct;

typedef struct
{
	double2  * h_v;
	double2  * h_v0;

	double   * h_vexc0;
	double   * h_vexc;

	double2  * h_density;
	double   * h_densityFrozen;

	double2  * d_v;
	double2  * d_v0;

	double   * d_vexc0;
	double   * d_vexc;

	double2 * d_density;
	double2 * d_density0;
	double  * d_densityFrozen;
	double  * d_test;

} hartreeStruct;

typedef struct
{
	// Device Pointers

	double2 * d_Z1;

} tempStruct;

typedef struct
{
	cublasHandle_t h_blas;
	cublasHandle_t d_blas;
	cusolverDnHandle_t solver;
	int     * d_devInfo;
	double  * d_workSpaceD;
	double2 * d_workSpaceZ;

} handleStruct;

#include "params.cuh"

int main (int argc, char * argv[])
{
	uint ldx, scandx, maxIter;

	cudaDeviceReset();

	h_checkCUDADevice ();

	// 1. Memory Allocation

	meshStruct mesh;

	mesh.h_r    = (double *) malloc (N_r * sizeof(double));
	mesh.h_P    = (double *) malloc (N_r * sizeof(double));
	mesh.h_drdx = (double *) malloc (N_r * sizeof(double));
	mesh.h_t    = (double *) malloc (N_t * sizeof(double));
	mesh.h_E    = (double *) malloc (N_t * sizeof(double));
	mesh.h_D1   = (double2 *) malloc (N_r * N_r * sizeof(double2));

	mesh.h_TD   = (double *) malloc (N_scan * sizeof(double));

	cudaMalloc ((void **) & mesh.d_r   , N_r * sizeof(double));
	cudaMalloc ((void **) & mesh.d_P   , N_r * sizeof(double));
	cudaMalloc ((void **) & mesh.d_drdx, N_r * sizeof(double));
	cudaMalloc ((void **) & mesh.d_t   , N_t * sizeof(double));
	cudaMalloc ((void **) & mesh.d_E   , N_t * sizeof(double));
	cudaMalloc ((void **) & mesh.d_D1  , N_r * N_r * sizeof(double2));

	propagatorsStruct propagators;

	// Monopole

	propagators.h_W   = (double2 **) malloc (N_l * sizeof(double2 *));
	propagators.h_d_W = (double2 **) malloc (N_l * sizeof(double2 *));

	cudaMalloc ((void **) & propagators.d_W, N_l * sizeof(double2 *));

	for (ldx = 0; ldx < N_l; ldx ++)
	{
		propagators.h_W[ldx] = (double2 *) malloc (N_r * N_r * sizeof(double2));
		cudaMalloc ((void **) & propagators.h_d_W[ldx], N_r * N_r * sizeof(double2));
	}

	cudaMemcpy (propagators.d_W, propagators.h_d_W, N_l * sizeof(double2 *), cudaMemcpyHostToDevice);

	// Dipole Propagator

	propagators.h_LPV   = (double **) malloc (N_m * sizeof(double *));
	propagators.h_d_LPV = (double **) malloc (N_m * sizeof(double *));

	propagators.h_LPD   = (double **) malloc (N_m * sizeof(double *));
	propagators.h_d_LPD = (double **) malloc (N_m * sizeof(double *));

	cudaMalloc ((void **) & propagators.d_LPV, N_m * sizeof(double *));

	cudaMalloc ((void **) & propagators.d_LPD, N_m * sizeof(double *));

	for (ldx = 0; ldx < N_m; ldx ++)
	{
		propagators.h_LPV[ldx] = (double *) malloc (N_l * N_l * sizeof(double));
		cudaMalloc ((void **) & propagators.h_d_LPV[ldx], N_l * N_l * sizeof(double));

		propagators.h_LPD[ldx] = (double *) malloc (N_l * sizeof(double));
		cudaMalloc ((void **) & propagators.h_d_LPD[ldx], N_l * sizeof(double));
	}

	cudaMemcpy (propagators.d_LPV, propagators.h_d_LPV, N_m * sizeof(double *), cudaMemcpyHostToDevice);
	cudaMemcpy (propagators.d_LPD, propagators.h_d_LPD, N_m * sizeof(double *), cudaMemcpyHostToDevice);

	// Quadrupole Propagator

	propagators.h_LDV   = (double **) malloc (N_m * sizeof(double *));
	propagators.h_d_LDV = (double **) malloc (N_m * sizeof(double *));

	propagators.h_LDD   = (double **) malloc (N_m * sizeof(double *));
	propagators.h_d_LDD = (double **) malloc (N_m * sizeof(double *));

	cudaMalloc ((void **) & propagators.d_LDV, N_m * sizeof(double *));

	cudaMalloc ((void **) & propagators.d_LDD, N_m * sizeof(double *));

	for (ldx = 0; ldx < N_m; ldx ++)
	{
		propagators.h_LDV[ldx] = (double *) malloc (N_l * N_l * sizeof(double));
		cudaMalloc ((void **) & propagators.h_d_LDV[ldx], N_l * N_l * sizeof(double));

		propagators.h_LDD[ldx] = (double *) malloc (N_l * sizeof(double));
		cudaMalloc ((void **) & propagators.h_d_LDD[ldx], N_l * sizeof(double));
	}

	cudaMemcpy (propagators.d_LDV, propagators.h_d_LDV, N_m * sizeof(double *), cudaMemcpyHostToDevice);
	cudaMemcpy (propagators.d_LDD, propagators.h_d_LDD, N_m * sizeof(double *), cudaMemcpyHostToDevice);

	// Octupole Propagator - P

	propagators.h_LFPV   = (double **) malloc (N_m * sizeof(double *));
	propagators.h_d_LFPV = (double **) malloc (N_m * sizeof(double *));

	propagators.h_LFPD   = (double **) malloc (N_m * sizeof(double *));
	propagators.h_d_LFPD = (double **) malloc (N_m * sizeof(double *));

	cudaMalloc ((void **) & propagators.d_LFPV, N_m * sizeof(double *));

	cudaMalloc ((void **) & propagators.d_LFPD, N_m * sizeof(double *));

	for (ldx = 0; ldx < N_m; ldx ++)
	{
		propagators.h_LFPV[ldx] = (double *) malloc (N_l * N_l * sizeof(double));
		cudaMalloc ((void **) & propagators.h_d_LFPV[ldx], N_l * N_l * sizeof(double));

		propagators.h_LFPD[ldx] = (double *) malloc (N_l * sizeof(double));
		cudaMalloc ((void **) & propagators.h_d_LFPD[ldx], N_l * sizeof(double));
	}

	cudaMemcpy (propagators.d_LFPV, propagators.h_d_LFPV, N_m * sizeof(double *), cudaMemcpyHostToDevice);
	cudaMemcpy (propagators.d_LFPD, propagators.h_d_LFPD, N_m * sizeof(double *), cudaMemcpyHostToDevice);

	// Octupole Propagator - F

	propagators.h_LFFV   = (double **) malloc (N_m * sizeof(double *));
	propagators.h_d_LFFV = (double **) malloc (N_m * sizeof(double *));

	propagators.h_LFFD   = (double **) malloc (N_m * sizeof(double *));
	propagators.h_d_LFFD = (double **) malloc (N_m * sizeof(double *));

	cudaMalloc ((void **) & propagators.d_LFFV, N_m * sizeof(double *));

	cudaMalloc ((void **) & propagators.d_LFFD, N_m * sizeof(double *));

	for (ldx = 0; ldx < N_m; ldx ++)
	{
		propagators.h_LFFV[ldx] = (double *) malloc (N_l * N_l * sizeof(double));
		cudaMalloc ((void **) & propagators.h_d_LFFV[ldx], N_l * N_l * sizeof(double));

		propagators.h_LFFD[ldx] = (double *) malloc (N_l * sizeof(double));
		cudaMalloc ((void **) & propagators.h_d_LFFD[ldx], N_l * sizeof(double));
	}

	cudaMemcpy (propagators.d_LFFV, propagators.h_d_LFFV, N_m * sizeof(double *), cudaMemcpyHostToDevice);
	cudaMemcpy (propagators.d_LFFD, propagators.h_d_LFFD, N_m * sizeof(double *), cudaMemcpyHostToDevice);

	// Set of Kohn-Sham Orbitals

	ksoSetTDStruct ksoSetTD;

	ksoSetTD.h_x   = (double2 **) malloc (N_l * sizeof(double2 *));
	ksoSetTD.h_d_x = (double2 **) malloc (N_l * sizeof(double2 *));

	ksoSetTD.h_xROT   = (double2 **) malloc (N_l * sizeof(double2 *));
	ksoSetTD.h_d_xROT = (double2 **) malloc (N_l * sizeof(double2 *));

	ksoSetTD.h_l   = (int *) malloc (N_kso * sizeof(int));
	ksoSetTD.h_m   = (int *) malloc (N_kso * sizeof(int));
	ksoSetTD.h_occ = (int *) malloc (N_kso * sizeof(int));
	ksoSetTD.h_density = (double2 *) malloc (N_har * N_r * sizeof(double2));

	cudaMalloc ((void **) & ksoSetTD.d_x, N_l * sizeof(double2 *));
	cudaMalloc ((void **) & ksoSetTD.d_xROT, N_l * sizeof(double2 *));

	cudaMalloc ((void **) & ksoSetTD.d_l, N_kso * sizeof(int));
	cudaMalloc ((void **) & ksoSetTD.d_m, N_kso * sizeof(int));
	cudaMalloc ((void **) & ksoSetTD.d_occ, N_kso * sizeof(int));
	cudaMalloc((void **) & ksoSetTD.d_density, N_har * N_r * sizeof(double2));

	for (ldx = 0; ldx < N_l; ldx ++)
	{
		ksoSetTD.h_x[ldx] = (double2 *) malloc (N_kso * N_r * sizeof(double2));
		ksoSetTD.h_xROT[ldx] = (double2 *) malloc (N_kso * N_r * sizeof(double2));
		cudaMalloc ((void **) & ksoSetTD.h_d_x[ldx], N_kso * N_r * sizeof(double2));
		cudaMalloc ((void **) & ksoSetTD.h_d_xROT[ldx], N_kso * N_r * sizeof(double2));
	}

	cudaMemcpy (ksoSetTD.d_x, ksoSetTD.h_d_x, N_l * sizeof(double2 *), cudaMemcpyHostToDevice);
	cudaMemcpy (ksoSetTD.d_xROT, ksoSetTD.h_d_xROT, N_l * sizeof(double2 *), cudaMemcpyHostToDevice);

	observableStruct observables;

	observables.h_dipoleMoment 		 = (double2 *) malloc (N_t * N_l * N_kso * sizeof(double2));
	observables.h_dipoleAcceleration = (double2 *) malloc (N_t * N_l * N_kso * sizeof(double2));
	observables.h_vhar         		 = (double2 *) malloc (N_t * N_l * N_kso * sizeof(double2));

	observables.h_vharDipole   		 = (double2 *) malloc (N_t * N_r * sizeof(double2));

	cudaMalloc((void **) & observables.d_vharDipole, N_t * N_r * sizeof(double2));

	hartreeStruct hartree;

	hartree.h_v 	  = (double2 *) malloc (N_har * N_r * sizeof(double2));
	hartree.h_v0 	  = (double2 *) malloc (N_r * sizeof(double2));
	hartree.h_vexc 	  = (double  *) malloc (N_r * sizeof(double));
	hartree.h_vexc0	  = (double  *) malloc (N_r * sizeof(double));
	hartree.h_density = (double2 *) malloc (N_har * N_r * sizeof(double2));
	hartree.h_densityFrozen = (double *) malloc (N_r * sizeof(double));

	cudaMalloc((void **) & hartree.d_density, N_har * N_r * sizeof(double2));
	cudaMalloc((void **) & hartree.d_density0, N_r * sizeof(double2));
	cudaMalloc((void **) & hartree.d_densityFrozen, N_r * sizeof(double));
	cudaMalloc((void **) & hartree.d_test, N_r * sizeof(double));
	cudaMalloc((void **) & hartree.d_v , N_har * N_r * sizeof(double2));
	cudaMalloc((void **) & hartree.d_v0, N_r * sizeof(double2));
	cudaMalloc((void **) & hartree.d_vexc, N_r * sizeof(double));
	cudaMalloc((void **) & hartree.d_vexc0, N_r * sizeof(double));

	tempStruct temp;

	cudaMalloc ((void **) & temp.d_Z1, N_r * sizeof(double2));

	// 1.X CUDA Library Handles

	handleStruct handles;

	cublasCreate (& handles.h_blas);
	cublasSetPointerMode(handles.h_blas, CUBLAS_POINTER_MODE_HOST);
	cublasCreate (& handles.d_blas);
	cublasSetPointerMode(handles.d_blas, CUBLAS_POINTER_MODE_DEVICE);
	cusolverDnCreate (& handles.solver);
	cudaMalloc ((void **) &  handles.d_devInfo, sizeof(int));
	handles.d_workSpaceD = NULL;
	handles.d_workSpaceZ = NULL;

	h_VRAMPrompt ();

	// 2. Load Working Data

	if (TD_SIMULATION == 1)
	{
		h_initialiseMesh (mesh);
		h_writeMesh (mesh);

		h_initialisePropagators (propagators);

		h_updateTimePropagation ();

		if (IN_SITU == 1)
		{
			maxIter = N_scan + 1;
		}
		else
		{
			maxIter = N_scan;
		}

		for (scandx = 0; scandx < maxIter; scandx ++)
		{
			printf("Scan = %d\n", scandx);
			h_initialiseTDKSOSet (ksoSetTD, hartree);

			if (scandx < N_scan)
			{
				h_initialiseField (mesh, scandx, 0);
			}
			else
			{
				h_initialiseField (mesh, scandx, 1);
			}

			h_timePropagation (ksoSetTD, hartree, propagators, mesh, observables, temp, handles, scandx);

			h_writeDipoleMoment (observables, scandx);

			// h_writeVHAR (observables, scandx);

			h_writeField (mesh, scandx);
		}
	}

	else
	{
		h_updateTimePropagation2 ();
	}

	free (mesh.h_r);
	free (mesh.h_P);
	free (mesh.h_drdx);
	free (mesh.h_t);
	free (mesh.h_E);
	free (mesh.h_D1);

	cudaFree (mesh.d_r);
	cudaFree (mesh.d_P);
	cudaFree (mesh.d_drdx);
	cudaFree (mesh.h_t);
	cudaFree (mesh.h_E);
	cudaFree (mesh.h_D1);

	for (ldx = 0; ldx < N_l; ldx ++)
	{
		free (propagators.h_W[ldx]);
		cudaFree(propagators.h_d_W[ldx]);

		free(ksoSetTD.h_x[ldx]);
		cudaFree(ksoSetTD.h_d_x[ldx]);

		free(ksoSetTD.h_xROT[ldx]);
		cudaFree(ksoSetTD.h_d_xROT[ldx]);
	}

	free (propagators.h_W);
	free (propagators.h_d_W);
	cudaFree (propagators.d_W);

	free (ksoSetTD.h_x);
	free (ksoSetTD.h_d_x);
	cudaFree (ksoSetTD.d_x);

	free (ksoSetTD.h_xROT);
	free (ksoSetTD.h_d_xROT);
	cudaFree (ksoSetTD.d_xROT);

	for (ldx = 0; ldx < N_m; ldx ++)
	{
		free (propagators.h_LPD[ldx]);
		cudaFree(propagators.h_d_LPD[ldx]);
		free (propagators.h_LPV[ldx]);
		cudaFree(propagators.h_d_LPV[ldx]);

		free (propagators.h_LDD[ldx]);
		cudaFree(propagators.h_d_LDD[ldx]);
		free (propagators.h_LDV[ldx]);
		cudaFree(propagators.h_d_LDV[ldx]);
	}

	free (propagators.h_LPD);
	free (propagators.h_d_LPD);
	cudaFree (propagators.d_LPD);

	free (propagators.h_LPV);
	free (propagators.h_d_LPV);
	cudaFree (propagators.d_LPV);

	free (propagators.h_LDD);
	free (propagators.h_d_LDD);
	cudaFree (propagators.d_LDD);

	free (propagators.h_LDV);
	free (propagators.h_d_LDV);
	cudaFree (propagators.d_LDV);

	free(ksoSetTD.h_l);
	free(ksoSetTD.h_m);
	free(ksoSetTD.h_occ);

	free(ksoSetTD.h_density);

	cudaFree(ksoSetTD.d_l);
	cudaFree(ksoSetTD.d_m);
	cudaFree(ksoSetTD.d_occ);
	cudaFree(ksoSetTD.d_density);

	free (observables.h_dipoleMoment);
	free (observables.h_dipoleAcceleration);
	free (observables.h_vhar);

	free (hartree.h_v);
	free (hartree.h_v0);
	free (hartree.h_vexc);
	free (hartree.h_vexc0);
	free (hartree.h_density);

	cudaFree (hartree.d_density);
	cudaFree (hartree.d_density0);
	cudaFree (hartree.d_v);
	cudaFree (hartree.d_v0);
	cudaFree (hartree.d_vexc);
	cudaFree (hartree.d_vexc0);

	cudaFree (temp.d_Z1);

	cublasDestroy(handles.h_blas);
	cublasDestroy(handles.d_blas);
	cusolverDnDestroy(handles.solver);
	cudaFree (handles.d_devInfo);
	cudaFree (handles.d_workSpaceD);
	cudaFree (handles.d_workSpaceZ);

	cudaDeviceReset();
}
