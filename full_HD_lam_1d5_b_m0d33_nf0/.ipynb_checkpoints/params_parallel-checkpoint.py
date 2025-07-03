"""
Parameter file for shell model.
"""
import sys
import numpy as np
import os
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# ################# Thin Layer (A0 = B0, H_3D conserved)

# def A1a(B0b,lam,alpha):
#     return -lam**(2*alpha)*B1c(B0b,lam,alpha)

# def A1b(B0a,lam,alpha):
#     return lam**(alpha)*A1c(B0a,lam, alpha)

# def A1c(B0a,lam, alpha):
#     return -B0a/(1+lam**alpha)

# def B1a(B0c,lam,alpha):
#     return lam**(alpha)*B1b(B0c,lam,alpha)

# def B1b(B0c,lam,alpha):
#     return -(1+lam**(alpha))**(-1)*B0c

# def B1c(B0b,lam,alpha):
#     return -B0b*(1-lam**(2*alpha))**(-1)

################# 2D MHD (A0 = -B0, A^2 conserved)

def A1a(B0b,lam,alpha):
    return -lam**(2*alpha)*B1c(B0b,lam,alpha)

def A1b(B0a,lam,alpha):
    return -lam**(alpha)*A1c(B0a,lam, alpha)

def A1c(B0a,lam, alpha):
    return -B0a/(1-lam**alpha)

def B1a(B0c,lam,alpha):
    return -lam**(alpha)*B1b(B0c,lam,alpha)

def B1b(B0c,lam,alpha):
    return -(1-lam**(alpha))**(-1)*B0c

def B1c(B0b,lam,alpha):
    return -B0b*(1-lam**(2*alpha))**(-1)

########################
### Parameter inputs ###
########################

### Numerical parameters:
nf = 4 # Forcing wavenumber
N = 24+nf # Number of wavenumbers 
lam = 1.5 # shell scaling
ii = 0
dt = 1e-2

### Model parameters
A0a = 1
A0b = -0.333333333333333333 #-1-np.sqrt(0.5)
A0 = np.array([A0a,A0b,-1-A0b])
B0 = -A0
alpha = -2 # Exponent of conserved quantity H_3D
A1 = [A1a(B0[1],lam,alpha),A1b(B0[0],lam,alpha),A1c(B0[0],lam, alpha)]
B1 = [B1a(B0[2],lam,alpha),B1b(B0[2],lam,alpha),B1c(B0[1],lam,alpha)]

eps0 = 1 # energy injection rate of u0 field
eps1 = 0
u0_init = 10**(-1.) #1e0
u1_init = 0.0
nu = 1e-7 # These are references from literature, not actual values. Check code
nu_l = 0 #0.001 # These are references from literature, not actual values. Check code
# For later use? 
Q = 0
# KE and ME spectral slopes:
alpha = -2/3. # KE
beta = 0.0 # ME

### Output parameters
H = 23.4 # Length of simulation in wall time (hours)
NS = 5 # Number of states to save in one run. 
NSc = 12000 # Iterations between 'energy balance' scalar saves.
NSc_avg = 10000000#300 # Iterations between adding to flux and spectra average, as well as histogram counts. # CHANGE AFTER STEADY STATE REACHED 
NSc_2 = np.nan#600 # Iterations between scalar saves (could be more frequent, for flux and thetas averages). If NSc_freq = np.nan, then it doesn't output this at all.
NSc_3 = np.nan#20 # Iterations between scalar saves (could be more frequent, for flux and thetas averages). If NSc_freq = np.nan, then it doesn't output this at all. # CHANGE AFTER STEADY STATE REACHED 
NSp = int(10000000/dt) # Avgeraging window. Iterations between flux and spectra average resets saves. Every time a state is saved, the average flux and spectra is saved and resets.

## For histogram
nbins = 30
bin_edges = np.linspace(-np.pi,np.pi,nbins+1)
bin_edges_centered = np.array([(bin_edges[i+1]+bin_edges[i])/2. for i in range(len(bin_edges)-1)])  #center of bins

### Parameter that changes per simulation
ps = [alpha] # alpha values
if len(ps)!=size:
    print("ERROR: number of parameters must match the number of cores!")

### 
# Are we continuing from a previous run?
overwrite = bool(1) # 1 if starting a new run, 0 if continuing from previous save. 
today = '2025-02-07'

######`##################################################################
### Do not change anything below this line ###

##########################
### Create directories ###
##########################
# Input directory
idirs = []
for p in ps:
    pstr = str(p).replace('.','d')
    dirname = './p_'+pstr+'/'
    idirs.append(dirname)
    # Root process makes the directory if it doesn't already exist
    if rank==0:
        if not os.path.exists(dirname):
            os.makedirs(dirname)
            
# Synchronizes parallel operations
comm.Barrier()

# Output directory
odirs = idirs
