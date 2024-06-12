
# RNN For Grid Connected Converter

This is a repository for my work using c++ to implement an RNN as a grid connected controller. The code is sitting in /main.

## Configuring Visual Studio:

I configured mine to use windows environment variables for my dependencies. Feel free to change this.

Edit the Project -> Project Properties -> Linker -> Input and replace Additional Dependencies with this:
```
	libopenblas.lib;msmpi.lib;$(CoreLibraryDependencies);%(AdditionalDependencies)
```

If machine is x64 machine:

- Edit the Project -> Project Properties -> C/C++ and replace Additional Include Directories with this:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;```$(MSMPI_INC);$(MSMPI_INC)\x64;$(ARMADILLO_INC);$(EIGEN_INC)```

- Edit the Project -> Project Properties -> Linker and replace Additional Library Directories with this:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;```$(MSMPI_LIB64);$(BLAS_LIB)```

- Edit the Project -> Project Properties -> Linker -> General and replace Additional Library Directories with this: 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;```$(MSMPI_LIB64);$(BLAS_LIB)```

if machine is x86:

- Edit the Project -> Project Properties -> C/C++ and replace Additional Include Directories with this: 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;```$(MSMPI_INC);$(MSMPI_INC)\x86;$(ARMADILLO_INC);$(EIGEN_INC)```

- Edit the Project -> Project Properties -> Linker and replace Additional Library Directories with this:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;```$(MSMPI_LIB32);$(BLAS_LIB)```

- Edit the Project -> Project Properties -> Linker -> Input and replace Additional Dependencies: 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;```libopenblas.lib;msmpi.lib;$(CoreLibraryDependencies);%(AdditionalDependencies)```

## Dependencies:

### BLAS:

Download OpenBLAS from here: https://github.com/xianyi/OpenBLAS/releases

Extract the zip file to ```C:\OpenBlas```

Edit your environment variables to the variable "OpenBLAS" to point to the lib directory inside this zip file. e.g. ```OpenBLAS = C:\OpenBLAS\OpenBLAS-0.3.21-x64\lib```

### MPI:
Install Microsoft MPI from here: https://learn.microsoft.com/en-us/message-passing-interface/microsoft-mpi?redirectedfrom=MSDN

The website will ask you to download ```msmpisetup.exe``` and ```msmpisdk.msi```. Download both

Run ```msmpisetup.exe``` and install Microsoft MPI into ```C:\Program Files\Microsoft MPI```.

Run ```msmpisdk.msi``` and install the MPI SDK into ```C:\Program Files (x86)\Microsoft SDKs\MPI\```

The installers will add the environment variables for MSMPI above.

### Armadillo:

Instructions and documentation: https://arma.sourceforge.net/download.html

Download the latest armadillo version, unzip and put somewhere like ```C:\Armadillo```

Add an environment variable ```ARMADILLO_INC``` pointing to the Include directory, e.g. ```C:\Armadillo\armadillo-11.4.3\include```

### Eigen:

Download Eigen latest (or 3.4.0) from here: https://eigen.tuxfamily.org/index.php?title=Main_Page

Put into filesystem somewhere like ```C:/Eigen```

Create new environment variable EIGEN_INC and put the path to the whole *unzipped* directory, e.g. ```EIGEN_INC=C:\Eigen\eigen-3.4.0```

## Running the program:

You must have ```mpiexec``` on the PATH, which should have happened automatically if you correctly installed Microsoft MPI.

Also, your runtime linker must be able to find ```libopenblas.dll```. I was having trouble with this, so I copied this dll from ```$OpenBLAS``` directory to the same directory that lmbp.exe is placed, e.g. ```$(SourceDir)/x64/Debug/```.

You can run the script like this: ```mpiexec -n $1 lmbp.exe $2```, where ```$1``` is the parameter for number of workers for the parallelization and ```$2``` is the parameter for how many multiples of 10 trajectories to run for training.

The simplest execution to test if everything is working is thus: ```mpiexec -n 1 lmbp.exe 1.```
