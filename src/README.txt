This program computes time-dependent strong-field dynamics in spherically
symmetric systems for linearly polarized fields using the formalism of
real-time time-dependent density functional theory.

Using this is easy!

You must first SSH into the CUDA server. To do this, open a terminal and
execute the following:

ssh graham@10.132.193.153

You will be prompted for a password, which is as follows:

bcd55c

Once you have logged into the server, type ggbSPH, which is a shortcut to the program directory.

The settings of the program can all be set in the "init" file. There are many possible variables
and parameters in the program. Default values are set for parameters not explicitly declared. The
default settings correspond to simulating HHG in hydrogen driven by a 20 fs 800 nm field of intensity
10^14 W/cm^2.

In the examples directory, there are copies of "init" files for each noble gas that I have
studied that you can use yourself. For each atom, there is also a figure of the HHG spectrum
calculated when using the example "init" file. You can simply copy the contents of the example
"init" file to the one used by the program in the /src directory to run the corresponding
simulation.

The simulation can be run for either a single scan, or for a pump-probe scheme. This is determined
by th "IN_SITU" variable.

All numerical output is saved in the output directory, which is sorted by data and simulation number, which
increases sequentially as multiple simulations are run. In order to keep track of multiple simulations,
three files acccompany the numerical data: (1) the init file used to define the simulation, (2) the params.cuh
which is determined by the init file and actually loaded by the program, and (3) the HHG spectrum from the simulation.

The program also generates figures:
	- Linear plots of the ground state radial wave functions
	- Contour plots of the ground state wave functions in (r, theta)
	- The radial mesh used
	- The derivative of the radial mesh with respect to memory index
	- The absorbing boundary
	- The time-dependent electric field
	- The dipole acceleration in both time and frequency
		- The total dipole acceleration
		- The dipole acceleration from each orbital
		- The dipole accleration from the active shells
	- The dipole moment in both time and frequency
		- The total dipole moment
		- The dipole moment from each orbital
		- The dipole moment from the active shells
	- The expectation value of the dipole Hartree potential for each orbital

The mathematics of how the program works is in the reference folder.
