"""
Algorithm for SOBP weights calculation, 
variation of the p-parameter, 
acquisition of the homogeneity ratio (HOM) and
design of the SOBP curve using Jette's method.

authors: I.S.L. Branco, A.L. Burin, J.J.N. Pereira,
P.T.D. Siqueira, J.M.B. Shorto, H. Yoriyaz

Please be sure to check the paper that provides
the basis for the discussion and analysis in this
script: 
https://doi.org/10.1016/j.radphyschem.2023.111043
"""

# %%_________Editable Part

# The original p-value obtained from the range-energy
# relationship of each MC code
p_org = 1.75 

# %%_________Import Packages
import os
import re
import sys
import inspect
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


# %%_________Functions

def sortFiles(fl):
    '''
    Sorts the files in a list by the incident energy value in their name,
    floating numbers have a "p" instead of a dot

    Parameters:
        fl (str): Full file path

    Returns:
        (float): Energy value in the file's name 
    '''

    return float(fl.rpartition("/")[-1].split("_")[-1].split(".")[0].replace("p","."))


def Jette_weights(p_value,n):
    '''
    Calculates the weight of the k-th beamlet to design an SOBP.
    
    Parameters:
        p_value (float): Power law parameter p.
        n (float): Number of energy intervals

    Returns:
        wgt (float): Array containing the beam weights
    '''
   
    # Defining the gamma variable to simplify the equation
    gam = 1-(1/p_value)
    
    # Declaration of an array to later store the weights
    wgt = np.zeros(n+1)
    
    # Calculating the weights according to paper's equation 3
    for k in range(n+1):
            
        if k == 0:
            wgt[k] = 1 - ((1 - (1/(2*n)))**gam)
        elif k == n:
            wgt[k] = (1/(2*n))**gam
        else:
            wgt[k] = (((1 - ((1/n)*(k-0.5)))**gam) -
                            ((1 - ((1/n)*(k+0.5)))**gam))
    return wgt


def hom_limitsdef(M,d):
    '''
    Calculates the boundaries of 80% of the SOBPs extension

    Parameters:
        M (float): Matrix M with all Bragg curves data
        d (float): List with SOBP's depth

    Returns:
        (int,int): Indexes of the limits in the SOBP's depth
    '''
    
    # Turn the depths list into an array
    depth=np.array(d)
    
    # Find the range limits of SOBP
    lim_ini = depth[M[:,0].argmax()]  # Initial range
    lim_end = depth[M[:,-1].argmax()] # Final range
    
    # Calculate the extension and important points of the SOBP
    sobp_ext = (lim_end - lim_ini)
    sobp_exthalf = sobp_ext/2
    sobp_middle = lim_end - sobp_exthalf
    
    # Calculate the SOBP limits within 80% of its extension
    new_lim_ini = sobp_middle - sobp_ext*0.4
    new_lim_end = sobp_middle + sobp_ext*0.4
    
    # Find the index of the depths near the new SOBPs limits 
    new_lim_ini_pos = abs(depth - new_lim_ini).argmin()  
    new_lim_end_pos = abs(depth - new_lim_end).argmin()
    
    return [new_lim_ini_pos,new_lim_end_pos+1]


# %%_________Get data from pristine Bragg Peaks

# This script's directory
# This command will only work if the code is executed 
# all at once (F5 in Spyder) and not in sections (F9 in Spyder)
# To bypass this command run the script from the downloaded folder
scpt_path = os.path.abspath(os.path.dirname(sys.argv[0]))

# Directory of the files with simulation data
sim_path = f'{scpt_path}/data_pristineBPs'

# List of files with simulation data
files = [f'{sim_path}/{f}' for f in os.listdir(sim_path) if '.txt' in f]
files.sort(key=sortFiles) # Organizes file names according to energy 


# %%_________Read the contents of the files with data from pristine Bragg
#            Peaks, one at a time, to create the matrix M (called Mmtx here)

# Loop to read one file at a time
for pos, f in enumerate(files):
    
    # Variables in which depth and deposited energy will be stored
    depth = list()
    ergdep = list()
    
    # Opens the txt file to extract the information, 
    # by reading one line after another
    with open(f) as fp:
        for i, line in enumerate(fp):
            if line[0].isnumeric():
                depth.append(float(line.split()[0]))
                ergdep.append(float(line.split()[1]))
    del(fp,i,line)
    
    # Checks if the variable Mmtx exists, if not, it is created
    if 'Mmtx' not in (globals() or locals()):
        Mmtx = np.zeros([len(ergdep),len(files)])
        
    # Store the results of the deposited energy in each one of the Mmtx columns    
    Mmtx[:,pos] = ergdep
del(pos,f)

# %%_________Calculation of the weights of each Bragg curve to build the SOBPs

# Create a p-vector ranging from 1.2 to 2.0, with 0.01 steps
p_vec = np.arange(1.2,2.01,0.01).round(2)

# Set the initial value of the homogeneity variable to zero
HOM_opt = 0

# Defines the indexes for 80% of the SOBP extension where
# the HOM parameter will be calculated 
lim_ini,lim_end = hom_limitsdef(M=Mmtx,
                                d=depth)

# Loop to select one p-value at a time
for p in p_vec:
    
    # Using the selected p value, compute the weight for each of the
    # 21 Bragg curves that will form the SOBP
    weights = Jette_weights(p_value=p,
                            n=Mmtx.shape[1]-1)
    
    # Multiply each column of Mmtx by the respective weight and sum them
    SOBP_Jette = (weights * Mmtx).sum(1)
    
    # Calculate the homogeneity ratio - HOM
    sobp = SOBP_Jette[lim_ini:lim_end] # SOBP plateau (80%)
    hom_aux = sobp.min()/sobp.max()    # Auxiliary HOM variable
    del(sobp)
    
    # Compare HOM stored value with the current auxiliary HOM analyzed
    if hom_aux > HOM_opt:
        
        p_opt = p                    # Stores the optimal p value
        HOM_opt = hom_aux            # Stores the optimal HOM value
        weights_opt = weights        # Stores the optimized weights
        SOBP_Jette_opt = SOBP_Jette  # Stores the optimized SOBP
    
    # Stores data from the p-original for comparison
    if p==p_org:
        
        HOM_org = hom_aux            # Stores the original HOM value
        weights_org = weights        # Stores the weights from p_org
        SOBP_Jette_org = SOBP_Jette  # Stores the SOBP from p_org
    
# %%_________Plot the results of SOBPs with p_original and p_optimized


# General figure structure
fig, axs = plt.subplots(2,1,figsize=(12,9),
                        gridspec_kw={'height_ratios': [2, 1],
                                      'hspace': 0.3})
plt.rcParams.update({'font.size': 18})

# Line graphs for optimal and original SOBPs
axs[0].plot(depth,SOBP_Jette_opt,
            color='mediumblue',
            alpha = 0.7,
            lw = 3,
            ls = '-',
            label = 'SOBP from p-optimal\n'+\
                rf'p$_{{{"opt"}}}$={p_opt}' + '\n' +\
                rf'HOM$_{{{"opt"}}}$={HOM_opt:.2f}')
axs[0].plot(depth,SOBP_Jette_org,
            color='brown',
            alpha = 0.5,
            lw = 3,
            ls = '--',
            label = 'SOBP from p-original\n'+\
                rf'p$_{{{"org"}}}$={p_org}' + '\n' +\
                rf'HOM$_{{{"org"}}}$={HOM_org:.2f}')


axs[0].set_xlim([0, depth[lim_end+50]])
axs[0].set_ylabel("Dose (a.u.)",labelpad = 10)
axs[0].set_xlabel("Depth (cm)",labelpad = 10)
axs[0].legend(fontsize=14)

axs[0].set_title("Dose distribution and weights obtained with Jette's method",
                 pad =8)
    
# Bar graph to show the weights adopted for SOBP with p-original weights
# and SOBP with p-optimal weights

# Setting arbitrary positions for bar graphs
width = 0.5

bpos_org = np.arange(0,
                  Mmtx.shape[1]*2.5*width,
                  2.5*width)
bpos_opt = bpos_org + width

bpos_tick = (bpos_org + bpos_opt)/2

# Bar graphs for optimal and original SOBPs weights
bars_org = axs[1].bar(bpos_org,
                  weights_org*100, # Weights are shown in %
                  color='brown',
                  edgecolor='k',
                  lw=1.5,
                  hatch='//',
                  align='center',
                  width=0.5,
                  alpha=0.8,
                  label=rf'Weights obtained with p$_{{{"org"}}}$')
axs[1].bar_label(bars_org,
                   fmt='%.1f',
                   fontsize=12,
                   padding=5,
                   rotation=90)

bars_opt = axs[1].bar(bpos_opt,
                  weights_opt*100, # Weights are shown in %
                  color='mediumblue',
                  edgecolor='k',
                  lw=1.5,
                  hatch='xx',
                  align='center',
                  width=0.5,
                  alpha=0.8,
                  label=rf'Weights obtained with p$_{{{"opt"}}}$')
axs[1].bar_label(bars_opt,
                   fmt='%.1f',
                   fontsize=12,
                   padding=5,
                   rotation=90)

axs[1].set_xticks(bpos_tick,
                  np.arange(Mmtx.shape[1])[::-1])


lim_up = (weights_opt*100).max() \
    if weights_opt.max() > weights_org.max() \
        else (weights_org*100).max()
        
axs[1].set_ylim([0,np.ceil(1.3*lim_up)])
axs[1].set_xlim([-width, bpos_opt[-1]+width])
axs[1].set_ylabel("Weights (%)",labelpad = 10)
axs[1].set_xlabel("Beamlets - k",labelpad = 10)
axs[1].legend(fontsize=14)




