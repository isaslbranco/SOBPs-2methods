# Algorithms for SOBP Weights Calculation

This repository contains two Python scripts that implement different algorithms to calculate the Spread-out Bragg Peaks (SOBP) and the Homogeneity Ratio (*HOM*). The algorithms design SOBP curves using the Jette's and MCMC methods to evaluate the weight of each pristine Bragg curve. In addition, instances of the MC codes output files were provided to aid the comprehension of the scripts.

Please consult the associated [original paper (ScienceDirect)](https://www.sciencedirect.com/science/article/pii/S0969806X23002888) for a detailed discussion and analysis of the methods, and cite it when using codes from this repository:
> Branco, I. S. L., Burin, A. L., Pereira, J. J. N., Siqueira, P. T. D., Shorto, J. M. B., & Yoriyaz, H. (2023). Comparison of methodologies for creating spread-out Bragg peaks in proton therapy using TOPAS and MCNP codes. Radiation Physics and Chemistry, 211, 111043. https://doi.org/10.1016/j.radphyschem.2023.111043

## Jette's Method
* **File**: `SOBP_Jette.py`
* **Description**: Jette's method designs the SOBP by considering the power law parameter (*p*) and the number of energy intervals (*n*). This script implements Jette's method to calculate the weights of the Bragg curves to compose an SOBP with a further *p-value* optimization step. This additional step explores a range of p-values to find one, defined as *p-optimal*, that results in the most homogeneous SOBP, determined by the *HOM* parameter.
* **Usage**: Run the script and generate a plot showing the SOBPs dose distribution with the original *p-value* and the *p-optimal*, along with the Bragg curve weights used in each SOBP.


## MCMC Method
* **File**: `SOBP_MCMC.py`
* **Description**: This script uses the MCMC method to determine the weights of the Bragg curves that will compose an SOBP. It begins by finding the maximum dose positions for each Bragg curve, constructs the D matrix, and defines the desired maximum doses. The algorithm then eliminates the negative weight curves and calculates new weights to compute the final SOBP.
* **Usage**: Run the script and generate a plot showing the dose distribution of the MCMC-based SOBP and the weights used for each Bragg curve.

## Data from 21 pristine Bragg curves
* **Folder**: `data_pristineBPs`
* **Description**: This folder contains instance data of 21 pristine Bragg curves. The curves are used as input in the algorithms previously mentioned to demonstrate the SOBP design for a maximum energy (E<sub>max</sub>) of 150 MeV and a width (χ) of 30%.


**Note**: Make sure to include the `SOBP_Jette.py` and `SOBP_MCMC.py` files along with the `data_pristineBPs` folder containing the instance data of the 21 pristine Bragg curves. Adjust the file paths as needed to ensure that the scripts can access the data in the folder.