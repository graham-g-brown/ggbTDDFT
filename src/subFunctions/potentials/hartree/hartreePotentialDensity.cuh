__global__
void d_getHartreePotentialDensity (
	hartreeStruct hartree,
	ksoSetTDStruct ksoSetTD
)
{
	int rdx = (blockIdx.x * blockDim.x) + threadIdx.x;
	int ldx = (blockIdx.y * blockDim.y) + threadIdx.y;

	int ksodx, lpdx, m;

	double2 temp, density;

	temp.x = 0.0;
	temp.y = 0.0;

	hartree.d_density[rdx + ldx * N_r].x = 0.0;
	hartree.d_density[rdx + ldx * N_r].y = 0.0;

	if (ldx == 0)
	{
		for (ksodx = 0; ksodx < N_kso; ksodx ++)
		{
			density.x = 0.0;
			density.y = 0.0;

			m = ksoSetTD.d_m[ksodx];

			for (lpdx = m; lpdx < N_l; lpdx ++)
			{
				density.x += pow(ksoSetTD.d_x[lpdx][rdx + ksodx * N_r].x, 2.0)
						   + pow(ksoSetTD.d_x[lpdx][rdx + ksodx * N_r].y, 2.0);
			}

			temp.x += double(ksoSetTD.d_occ[ksodx]) * density.x;
		}
	}

	else if (ldx == 1)
	{
		for (ksodx = 0; ksodx < N_kso; ksodx ++)
		{
			density.x = 0.0;
			density.y = 0.0;

			m = ksoSetTD.d_m[ksodx];

			lpdx = m;

			density.x += d_cu (lpdx, m) * (ksoSetTD.d_x[lpdx + 1][rdx + ksodx * N_r].x * ksoSetTD.d_x[lpdx][rdx + ksodx * N_r].x
									    +  ksoSetTD.d_x[lpdx + 1][rdx + ksodx * N_r].y * ksoSetTD.d_x[lpdx][rdx + ksodx * N_r].y);

			for (lpdx = m + 1; lpdx < N_l - 1; lpdx ++)
			{
				density.x += d_cu (lpdx, m) * (ksoSetTD.d_x[lpdx + 1][rdx + ksodx * N_r].x * ksoSetTD.d_x[lpdx][rdx + ksodx * N_r].x
										    +  ksoSetTD.d_x[lpdx + 1][rdx + ksodx * N_r].y * ksoSetTD.d_x[lpdx][rdx + ksodx * N_r].y)

						   + d_cd (lpdx, m) * (ksoSetTD.d_x[lpdx - 1][rdx + ksodx * N_r].x * ksoSetTD.d_x[lpdx][rdx + ksodx * N_r].x
							   			    +  ksoSetTD.d_x[lpdx - 1][rdx + ksodx * N_r].y * ksoSetTD.d_x[lpdx][rdx + ksodx * N_r].y);
			}

			lpdx = N_l - 1;

			density.x += d_cd (lpdx, m) * (ksoSetTD.d_x[lpdx - 1][rdx + ksodx * N_r].x * ksoSetTD.d_x[lpdx][rdx + ksodx * N_r].x
									    +  ksoSetTD.d_x[lpdx - 1][rdx + ksodx * N_r].y * ksoSetTD.d_x[lpdx][rdx + ksodx * N_r].y);

			temp.x += double(ksoSetTD.d_occ[ksodx]) * density.x;
		}
	}

	else if (ldx == 2)
	{
		for (ksodx = 0; ksodx < N_kso; ksodx ++)
		{
			density.x = 0.0;
			density.y = 0.0;

			m = ksoSetTD.d_m[ksodx];

			for (lpdx = m; lpdx < m + 2; lpdx ++)
			{
				density.x += d_qu (lpdx, m) * (ksoSetTD.d_x[lpdx + 2][rdx + ksodx * N_r].x * ksoSetTD.d_x[lpdx][rdx + ksodx * N_r].x
										    +  ksoSetTD.d_x[lpdx + 2][rdx + ksodx * N_r].y * ksoSetTD.d_x[lpdx][rdx + ksodx * N_r].y)

						   + d_p  (lpdx, m) * (pow(ksoSetTD.d_x[lpdx][rdx + ksodx * N_r].x, 2.0)
						   				    +  pow(ksoSetTD.d_x[lpdx][rdx + ksodx * N_r].y, 2.0));
			}

			for (lpdx = m + 2; lpdx < N_l - 2; lpdx ++)
			{
				density.x += d_qu (lpdx, m) * (ksoSetTD.d_x[lpdx + 2][rdx + ksodx * N_r].x * ksoSetTD.d_x[lpdx][rdx + ksodx * N_r].x
										    +  ksoSetTD.d_x[lpdx + 2][rdx + ksodx * N_r].y * ksoSetTD.d_x[lpdx][rdx + ksodx * N_r].y)

						   + d_qd (lpdx, m) * (ksoSetTD.d_x[lpdx - 2][rdx + ksodx * N_r].x * ksoSetTD.d_x[lpdx][rdx + ksodx * N_r].x
											+  ksoSetTD.d_x[lpdx - 2][rdx + ksodx * N_r].y * ksoSetTD.d_x[lpdx][rdx + ksodx * N_r].y)

						   + d_p  (lpdx, m) * (pow(ksoSetTD.d_x[lpdx][rdx + ksodx * N_r].x, 2.0) + pow(ksoSetTD.d_x[lpdx][rdx + ksodx * N_r].y, 2.0));
			}

			for (lpdx = N_l - 2; lpdx < N_l; lpdx ++)
			{
				density.x += d_qd (lpdx, m) * (ksoSetTD.d_x[lpdx - 2][rdx + ksodx * N_r].x * ksoSetTD.d_x[lpdx][rdx + ksodx * N_r].x
											+  ksoSetTD.d_x[lpdx - 2][rdx + ksodx * N_r].y * ksoSetTD.d_x[lpdx][rdx + ksodx * N_r].y)

						   + d_p  (lpdx, m) * (pow(ksoSetTD.d_x[lpdx][rdx + ksodx * N_r].x, 2.0) + pow(ksoSetTD.d_x[lpdx][rdx + ksodx * N_r].y, 2.0));
			}

			temp.x += double(ksoSetTD.d_occ[ksodx]) * density.x;
		}
	}

	else if (ldx == 3)
	{
		for (ksodx = 0; ksodx < N_kso; ksodx ++)
		{
			density.x = 0.0;
			density.y = 0.0;

			m = ksoSetTD.d_m[ksodx];

			lpdx = m;

			density.x += d_p3lm (lpdx, m) * (ksoSetTD.d_x[lpdx + 1][rdx + ksodx * N_r].x * ksoSetTD.d_x[lpdx][rdx + ksodx * N_r].x
										  +  ksoSetTD.d_x[lpdx + 1][rdx + ksodx * N_r].y * ksoSetTD.d_x[lpdx][rdx + ksodx * N_r].y)

					   + d_f3lm (lpdx, m) * (ksoSetTD.d_x[lpdx + 3][rdx + ksodx * N_r].x * ksoSetTD.d_x[lpdx][rdx + ksodx * N_r].x
		   								  +  ksoSetTD.d_x[lpdx + 3][rdx + ksodx * N_r].y * ksoSetTD.d_x[lpdx][rdx + ksodx * N_r].y);

			for (lpdx = m + 1; lpdx < m + 3; lpdx ++)
			{
				density.x += d_p3lm (lpdx - 1, m) * (ksoSetTD.d_x[lpdx - 1][rdx + ksodx * N_r].x * ksoSetTD.d_x[lpdx][rdx + ksodx * N_r].x
											      +  ksoSetTD.d_x[lpdx - 1][rdx + ksodx * N_r].y * ksoSetTD.d_x[lpdx][rdx + ksodx * N_r].y)

						   + d_p3lm (lpdx, m) * (ksoSetTD.d_x[lpdx + 1][rdx + ksodx * N_r].x * ksoSetTD.d_x[lpdx][rdx + ksodx * N_r].x
											  +  ksoSetTD.d_x[lpdx + 1][rdx + ksodx * N_r].y * ksoSetTD.d_x[lpdx][rdx + ksodx * N_r].y)

						   + d_f3lm (lpdx, m) * (ksoSetTD.d_x[lpdx + 3][rdx + ksodx * N_r].x * ksoSetTD.d_x[lpdx][rdx + ksodx * N_r].x
			   								  +  ksoSetTD.d_x[lpdx + 3][rdx + ksodx * N_r].y * ksoSetTD.d_x[lpdx][rdx + ksodx * N_r].y);
			}

			for (lpdx = m + 3; lpdx < N_l - 3; lpdx ++)
			{
				density.x +=

							d_f3lm (lpdx - 3, m) * (ksoSetTD.d_x[lpdx - 3][rdx + ksodx * N_r].x * ksoSetTD.d_x[lpdx][rdx + ksodx * N_r].x
								   			      +  ksoSetTD.d_x[lpdx - 3][rdx + ksodx * N_r].y * ksoSetTD.d_x[lpdx][rdx + ksodx * N_r].y)

						   + d_p3lm (lpdx - 1, m) * (ksoSetTD.d_x[lpdx - 1][rdx + ksodx * N_r].x * ksoSetTD.d_x[lpdx][rdx + ksodx * N_r].x
											      +  ksoSetTD.d_x[lpdx - 1][rdx + ksodx * N_r].y * ksoSetTD.d_x[lpdx][rdx + ksodx * N_r].y)

						   + d_p3lm (lpdx, m) * (ksoSetTD.d_x[lpdx + 1][rdx + ksodx * N_r].x * ksoSetTD.d_x[lpdx][rdx + ksodx * N_r].x
											  +  ksoSetTD.d_x[lpdx + 1][rdx + ksodx * N_r].y * ksoSetTD.d_x[lpdx][rdx + ksodx * N_r].y)

						   + d_f3lm (lpdx, m) * (ksoSetTD.d_x[lpdx + 3][rdx + ksodx * N_r].x * ksoSetTD.d_x[lpdx][rdx + ksodx * N_r].x
			   								  +  ksoSetTD.d_x[lpdx + 3][rdx + ksodx * N_r].y * ksoSetTD.d_x[lpdx][rdx + ksodx * N_r].y);
			}

			for (lpdx = N_l - 4; lpdx < N_l - 1; lpdx ++)
			{
				density.x += d_f3lm (lpdx - 3, m) * (ksoSetTD.d_x[lpdx - 3][rdx + ksodx * N_r].x * ksoSetTD.d_x[lpdx][rdx + ksodx * N_r].x
								   			      +  ksoSetTD.d_x[lpdx - 3][rdx + ksodx * N_r].y * ksoSetTD.d_x[lpdx][rdx + ksodx * N_r].y)

						   + d_p3lm (lpdx - 1, m) * (ksoSetTD.d_x[lpdx - 1][rdx + ksodx * N_r].x * ksoSetTD.d_x[lpdx][rdx + ksodx * N_r].x
											      +  ksoSetTD.d_x[lpdx - 1][rdx + ksodx * N_r].y * ksoSetTD.d_x[lpdx][rdx + ksodx * N_r].y)

						   + d_p3lm (lpdx, m) * (ksoSetTD.d_x[lpdx + 1][rdx + ksodx * N_r].x * ksoSetTD.d_x[lpdx][rdx + ksodx * N_r].x
											  +  ksoSetTD.d_x[lpdx + 1][rdx + ksodx * N_r].y * ksoSetTD.d_x[lpdx][rdx + ksodx * N_r].y);
			}

			density.x += d_f3lm (lpdx - 3, m) * (ksoSetTD.d_x[lpdx - 3][rdx + ksodx * N_r].x * ksoSetTD.d_x[lpdx][rdx + ksodx * N_r].x
											  +  ksoSetTD.d_x[lpdx - 3][rdx + ksodx * N_r].y * ksoSetTD.d_x[lpdx][rdx + ksodx * N_r].y)

					   + d_p3lm (lpdx - 1, m) * (ksoSetTD.d_x[lpdx - 1][rdx + ksodx * N_r].x * ksoSetTD.d_x[lpdx][rdx + ksodx * N_r].x
											  +  ksoSetTD.d_x[lpdx - 1][rdx + ksodx * N_r].y * ksoSetTD.d_x[lpdx][rdx + ksodx * N_r].y);

			temp.x += double(ksoSetTD.d_occ[ksodx]) * density.x;
		}
	}

	hartree.d_density[rdx + ldx * N_r] = temp;
}

__global__
void d_getHartreePotentialDensity0 (
	hartreeStruct hartree,
	ksoSetTDStruct ksoSetTD
)
{
	int rdx = (blockIdx.x * blockDim.x) + threadIdx.x;
	int ldx = (blockIdx.y * blockDim.y) + threadIdx.y;

	int ksodx, lpdx, m;

	double2 density, temp;

	temp.x = 0.0;
	temp.y = 0.0;

	hartree.d_density[rdx].x = 0.0;
	hartree.d_density[rdx].y = 0.0;

	if (ldx == 0)
	{
		for (ksodx = 0; ksodx < N_kso; ksodx ++)
		{
			density.x = 0.0;
			density.y = 0.0;

			m = ksoSetTD.d_m[ksodx];

			for (lpdx = m; lpdx < N_l; lpdx ++)
			{
				density.x += pow(ksoSetTD.d_x[lpdx][rdx + ksodx * N_r].x, 2.0)
						   + pow(ksoSetTD.d_x[lpdx][rdx + ksodx * N_r].y, 2.0);
			}

			temp.x += double(ksoSetTD.d_occ[ksodx]) * density.x;
		}

		hartree.d_density0[rdx + ldx * N_r] = temp;
	}
}
