# Naming conventions

## HDF5 file names
The averaged statistics are stored in HDF5 files, which come in two types:

1. chan_l[*number1*]_L3xL2xL1.h5
2. chan_hxsx[*number2*]e-3_L3xL2xL1.h5

In these file names, the prefix "chan" refers to the openchannelflow setup of the simulations. The dimensions L1, L2, and L3 correspond to the streamwise, cross-stream, and vertical extent of the computational domain, respectively.

For configurations where s1 equals s2 (please refer to Figure 1 of the manuscript for the definitions of s1 and s2), the naming follows the first convention. The symbol "l" represents the length of the repeating unit, and *number1* represents the specific length value. In this study, *number1* can be 2, 4, 6, or 12.

On the other hand, configurations that utilize boundary layer height-based scaling are named according to the second convention, as s1 is not equal to s2. These configurations satisfy the condition h1/s1 = h2/s2. In the file names, "hxsx" denotes the ratio h1/s1. *number2* represents the value of this ratio multiplied by 1000 to avoid decimal numbers. To indicate this operation, *number2* is followed by "e-3". In this study, *number2* can take the values 500, 250, 167, or 83.

### Additional prefixes
The HDF5 files may also have an additional prefix of "ruu," "rvv," or "ruw." The "r" in these prefixes signifies that these files store two-point correlation values. The two variables that follow represent the fields for which the correlation is computed. For instance, a file with the prefix "ruw" would contain the two-point correlation between the streamwise velocity (u) and the vertical velocity (w).

### Additional suffixes
In this study, each simulation was initially conducted for 200 large eddy turnover times. However, certain larger simulations were divided into four batches, each consisting of 50 large eddy turnovers. The averaged values from these simulations are distinguished by the suffixes 'restart[*number*],' where 'number' ranges from 0 to 3.

Due to insufficient convergence during the initial 200 large eddy turnover times, some simulations required an additional 200 large eddy turnover times. These subsequent runs are indicated by the suffix 'run2'. 

## Naming convention in post-processing
Each Jupyter Notebook script typically consists of four dictionaries:

1. `f`: This dictionary is responsible for loading all the variables from HDF5 files.
2. `temp`: This dictionary is used to store the temporal stress values calculated from the variables loaded in `f`.
3. `disp`: This dictionary is used to store the dispersive stress values computed from the variables loaded in `f`.
4. `stress`: This dictionary is used to store the total stress values, which are the sum of `temp` and `disp`.

The keys of these dictionaries follow the syntax *field* _ *simulation*, where the field represents a first or second order field (e.g., u, uv, etc.), and simulation corresponds to the HDF5 file name with the prefix 'chan_' removed, as well as the '.h5' extension.