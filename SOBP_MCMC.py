"""
Algorithm for SOBP weights calculation, 
acquisition of the homogeneity ratio (HOM) and
design of the SOBP curve using MCMC method.

authors: I.S.L. Branco, A.L. Burin, J.J.N. Pereira,
P.T.D. Siqueira, J.M.B. Shorto, H. Yoriyaz

Please be sure to check the paper that provides
the basis for the discussion and analysis in this
script: 
https://doi.org/10.1016/j.radphyschem.2023.111043
"""

# %%_________Import Packages
import os
import re
import sys
import inspect
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# os.chdir('/home/isabela/Python/Projetos/SOBP_GitHFolder')
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
# Sorts the file names according to the energy,
# from the highest to the lowest
files.sort(key=sortFiles, reverse=True) 


# %%_________Read the contents of the files with data from pristine Bragg
#            Peaks, one at a time, to create the matrix M (called Mmtx here)

# Loop to read one file at a time
for pos_f, f in enumerate(files):
    
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
    Mmtx[:,pos_f] = ergdep
del(pos_f,f)

# Create a variable to control the number of curves
c = 0

# %%_________Calculation of the weights of each Bragg curve to build the SOBPs

# Find the maximum dose position for each one of the curves
# in Mmtx and save them in a vector
pos = Mmtx.argmax(0)

# Create the D matrix
D = Mmtx[pos,:]

# Set the vector dmax with the intended maximum doses
# Here the number 1 indicates 100% of the dose
dmax = np.repeat(1.0, Mmtx.shape[1])

# The first and last values have been adjusted to reduce
# the spikes at the ends of the SOBPs, we encourage you 
# to change these values and observe the differences in
# the final SOBP (SOBP_MCMC)
dmax[0] = 0.97
dmax[-1] = 0.99

# Invert the D matrix and calculate the initial weights
w_ini = np.linalg.inv(D).dot(dmax)

# %%_________Correcting negative weights, if any

# Store the position of negative weights
wpos_neg = np.where(w_ini<0)[0]

# The following steps ensure the SOBP has the correct
# width (Chi value) in a way that the lowest and highest
# energy Bragg curves are always included, without 
# adopting negative weights, and thus preventing their deletion

# If the weight of the higher energy curve E0 is negative (see w_ini[0]),  
# add one position to all values in wpos_neg
## Warning: if one position has been added to all
## values in wpos_neg, the lowest energy curve will
## be indicated here as in position 21 instead of 20
if wpos_neg.any() and wpos_neg[0] == 0:
    wpos_neg += 1 
    
    # Corrects the last position of wpos_neg if it is larger 
    # (i.e. here, 21) than the last position (here, 20) of the
    # weight vector (w_ini)
    if wpos_neg[-1] == len(w_ini):
        wpos_neg = wpos_neg[:-1]
        
# If the weight of the lowest energy curve is negative
# (here E20, i.e, if wpos_neg[-1]==20), delete its position
# in wpos_neg
if wpos_neg.any() and wpos_neg[-1] == len(w_ini)-1:
    wpos_neg = wpos_neg[:-1]
    
    # Adds the position of the second lowest energy
    # curve (E19) if it is not already in wpos_neg 
    if len(w_ini)-2 not in wpos_neg:
        wpos_neg = np.insert(wpos_neg,len(wpos_neg), len(w_ini)-2)

# New variables were created to manipulate their values
# in case of negative weights that needed to be deleted,
# which also affected their values.
## Copy of the results for each Bragg curve
Mmtx_mod = Mmtx.copy() 
## Copy of the D matrix
D_mod = D.copy() 
## Copy of the dmax vector 
dmax_mod = dmax.copy() 
## Copy of weigth vector just used as a initial reference
wm = w_ini.copy() 
## Vector that will be altered to show the position of the
## curves with positive weights (i.e., that were not excluded)
w_left = np.arange(len(w_ini)) 

# Loop that eliminates negative weight curves
while (wm < 0).any() :
    
    # Control variable to select each curve with negative weight
    c += 1
    
    # Position of the curve with negative weight, taking into
    # account those that have already been deleted
    ncurv = wpos_neg[c-1] - (c-1)
    
    # Deletes all points of the Bragg curve (column) relative
    # to the negative weight position 
    Mmtx_mod = np.delete(Mmtx_mod, ncurv, 1)
    
    # Deletes row and column in D matrix relative
    # to the negative weight position 
    D_mod = np.delete(np.delete(D_mod, ncurv, 0), ncurv, 1)
    
    # Deletes the position value in the dmax vector
    # relative to the examined negative weight
    dmax_mod = np.delete(dmax_mod, ncurv, 0)
    
    # Excludes the negative weight position, so that
    # only the positions of the original positive weights
    # remain at the very end
    w_left = np.delete(w_left, ncurv, 0)
    
    # Recalculates the weights with the remaining curves
    # in the evaluation.
    ## The loop is interrupted when all weight values in wm
    ## are positive
    wm = np.linalg.inv(D_mod).dot(dmax_mod)

# Transforms the weight contribution to percentages
wm_perc = (wm/np.sum(wm))*100

# Multiply each column of Mmtx by the respective weight and sum them
SOBP_MCMC = ( wm_perc * Mmtx_mod).sum(1)   

# Defines the indexes for 80% of the SOBP extension where
# the HOM parameter will be calculated 
lim_end,lim_ini = hom_limitsdef(M=Mmtx_mod,
                                d=depth)

# Calculate the homogeneity ratio - HOM
sobp = SOBP_MCMC[lim_ini:lim_end] # SOBP plateau (80%)
HOM = sobp.min()/sobp.max()    # Auxiliary HOM variable
del(sobp)

# %%_________Plot the results of SOBPs with p_original and p_optimized

# General figure structure
fig, axs = plt.subplots(2,1,figsize=(12,9),
                        gridspec_kw={'height_ratios': [2, 1],
                                      'hspace': 0.3})
plt.rcParams.update({'font.size': 18})

# Line graph for MCMC SOBPs
axs[0].plot(depth,SOBP_MCMC,
            color='forestgreen',
            alpha = 0.7,
            lw = 3,
            ls = '-',
            label = 'MCMC SOBP\n'+\
                f'HOM={HOM:.2f}')

axs[0].set_xlim([0, depth[lim_end+50]])
axs[0].set_ylabel("Dose (a.u.)",labelpad = 10)
axs[0].set_xlabel("Depth (cm)",labelpad = 10)
axs[0].legend(fontsize=14)

axs[0].set_title("Dose distribution and weights obtained with MCMC method",
                 pad =8)
    
# Bar graph to show the weights adopted for MCMC SOBP 
## Setting arbitrary positions for bar graphs
width = 0.5

bpos = np.arange(0,
                  Mmtx.shape[1]*1.5*width,
                  1.5*width)


bpos_tick = bpos

## Bar graphs for SOBPs weights according to the energy of
## each Bragg curve
bars = axs[1].bar(bpos[::-1][w_left],
                  wm_perc, # Weights are shown in %
                  color='green',
                  edgecolor='k',
                  lw=1.5,
                  hatch='//',
                  align='center',
                  width=0.5,
                  alpha=0.8,
                  label='Weights obtained with MCMC Method')
axs[1].bar_label(bars,
                   fmt='%.1f',
                   fontsize=12,
                   padding=5,
                   rotation=90)

axs[1].set_xticks(bpos_tick[::-1][w_left],
                  w_left)
axs[1].set_ylim([0,np.ceil(1.3*wm_perc.max())])
axs[1].set_xlim([-width, bpos[-1]+width])
axs[1].set_ylabel("Weights (%)",labelpad = 10)
axs[1].set_xlabel("Beamlets - k",labelpad = 10)
axs[1].legend(fontsize=14,
              loc='upper left')




