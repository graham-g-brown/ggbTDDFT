__device__
double d_p3lm (
	int ll,
	int mm
)
{
	if (ll >= mm)
	{
		double l = double(ll);
		double m = double(mm);

		return 1.50 * (l * (l + 2.0) - 5.0 * pow(m, 2.0)) / (2.0 * l - 1.0) / (2.0 * l + 5.0) * sqrt((pow(l + 1.0, 2.0) - pow(m, 2.0)) / (2.0 * l + 1.0) / (2.0 * l + 3.0));
	}
	else
	{
		return 0.0;
	}
}

__device__
double d_f3lm (
	int ll,
	int mm
)
{
	if (ll >= mm)
	{
		double l = double(ll);
		double m = double(mm);

		return 2.50 / (2.0 * l + 3.0) / (2.0 * l + 5.0) * sqrt((pow(l + 1.0, 2.0) - pow(m, 2.0)) * (pow(l + 2.0, 2.0) - pow(m, 2.0)) * (pow(l + 3.0, 2.0) - pow(m, 2.0)) / (2.0 * l + 1.0) / (2.0 * l + 7.0));
	}
	else
	{
		return 0.0;
	}
}

__device__
double d_c (
	int ll,
	int mm
)
{
	if ((mm <= ll) && (ll >= 0))
	{
		double l = double(ll);
		double m = double(mm);

		return sqrt( (pow(l + 1.0, 2.0) - pow(m, 2.0)) / ( (2.0 * l + 1.0) * (2.0 * l + 3.0) ) );
	}
	else
	{
		return 0.0;
	}
}

__device__
double d_cu (
	int ll,
	int mm
)
{
	if ((ll >= 0) && (ll >= mm))
	{
		double l = double(ll);
		double m = double(mm);

		return sqrt( (pow(l + 1.0, 2.0) - pow(m, 2.0)) / ( (2.0 * l + 1.0) * (2.0 * l + 3.0) ) );
	}
	else
	{
		return 0.0;
	}
}

__device__
double d_cd (
	int ll,
	int mm
)
{
	if ((ll >= 1) && (ll - 1 >= mm))
	{
		double l = double(ll);
		double m = double(mm);

		return sqrt( (pow(l, 2.0) - pow(m, 2.0)) / ( (2.0 * l + 1.0) * (2.0 * l - 1.0) ) );
	}
	else
	{
		return 0.0;
	}
}

__device__
double d_p (
	int ll,
	int mm
)
{
	if ((mm <= ll) && (ll >= 1))
	{
		double p;

		double l = double(ll);
		double m = double(mm);

		p = (l * (l + 1.0) - 3.0 * pow(m, 2.0)) / ((2.0 * l - 1.0) * (2.0 * l + 3.0));

		return p;
	}
	else
	{
		return 0.0;
	}
}

__device__
double d_q (
	int ll,
	int mm
)
{
	if ((mm <= ll) && (ll >= 0))
	{
		double l = double(ll);
		double m = double(mm);

		return 1.50 / (2.0 * l + 3.0) * sqrt( (pow(l + 1.0, 2.0) - pow(m, 2.0)) * (pow(l + 2.0, 2.0) - pow(m, 2.0)) / ( (2.0 * l + 1.0) * (2.0 * l + 5.0) ) );
	}
	else
	{
		return 0.0;
	}
}

__device__
double d_qu (
	int ll,
	int mm
)
{
	if ((mm <= ll) && (ll >= 0))
	{
		double l = double(ll);
		double m = double(mm);

		return 1.50 / (2.0 * l + 3.0) * sqrt( (pow(l + 1.0, 2.0) - pow(m, 2.0)) * (pow(l + 2.0, 2.0) - pow(m, 2.0)) / ( (2.0 * l + 1.0) * (2.0 * l + 5.0) ) );
	}
	else
	{
		return 0.0;
	}
}

__device__
double d_qd (
	int ll,
	int mm
)
{
	if ((mm <= ll - 2) && (ll >= 2))
	{
		double l = double(ll);
		double m = double(mm);

		return 1.50 / (2.0 * l - 1.0) * sqrt( (pow(l, 2.0) - pow(m, 2.0)) * (pow(l - 1.0, 2.0) - pow(m, 2.0)) / ( (2.0 * l + 1.0) * (2.0 * l - 3.0) ) );
	}
	else
	{
		return 0.0;
	}
}
