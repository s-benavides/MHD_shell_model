import sys
import model as shell
import pathlib
import numpy as np
import h5py
import time
from mpi4py import MPI
from scipy.stats import binned_statistic
import glob as glob

from os import path
import os

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

from params_parallel import *

# CREATE RUN FILE. Can be manually deleted if you want to 'kill' the run softly. Checked for every 11 mins.
if rank==0:
    with open('RUNNING.txt', 'w') as creating_new_csv_file: 
       pass 
    print("Empty RUNNING File Created Successfully",flush=True)

# Process decides which directories and parameter values to work on:
# PARAMETER TO CHANGE
alpha = ps[rank]
odir = odirs[rank]
idir = idirs[rank]

# Initialize
run = shell.shell(nf,N,dt,lam,A0,B0,A1,B1,eps0,eps1,Q,nu,nu_l,u0_init,u1_init,alpha=alpha,beta=beta)

state_file = pathlib.Path(idir+run.export_name(today)+"_state.h5").exists()

if not state_file:
    overwrite = True

# Printing to file:
if overwrite:
    log = open(odir+'log.txt','w')
else:
    log = open(odir+'log.txt','a')
sys.stdout = log

print(str({'nf':nf, 'N':N,'lam':lam, 'dt':dt,'A0':A0,'B0':B0,'A1':A1,'B1':B1,'eps0':eps0,'eps1':eps1,'alpha':alpha,'beta':beta,'u0_init':u0_init,'u1_init':u1_init,'Q':Q,'H':H,'NS':NS,'NSp':NSp,'NSc':NSc,'NSc_2':NSc_2,'NSc_3':NSc_3,'overwrite':overwrite,'idir':idir,'odir':odir,'rank':rank}),flush=True)

if not state_file:
    print("No state file found. Starting from scratch.", flush=True)

print('Initializing...',flush=True)

# For flux calculation in the code
r3D = np.log(-run.A1[0]/run.B1[2])/np.log(run.lam)/2

if not overwrite:
    # Load data:
    # Set state from last run
    print("Loading data.")
    run.load_data(idir+run.export_name(today)+"_state.h5")
    print("Starting at sim time = %.3e" % run.t,flush=True)
    
    # Load scalar data to keep appending to
    odata = []
    # Check if file exists
    if path.exists(idir+run.export_name(today)+"_scalars.h5"):
        with h5py.File(idir+run.export_name(today)+"_scalars.h5",'r') as f:
            odata = []
            for key in run.okeys:
                odata.append(f['scalars'][key][:])
        
            odata = list(np.array(odata).T)
        print("Adding to previous scalars file.",flush=True)
    else:
        print("Making new scalars file",flush=True)

    if (~np.isnan(NSc_2)):
        # Load scalar data to keep appending to
        odata_2 = []
        # Check if file exists
        if path.exists(idir+run.export_name(today)+"_scalars_2.h5"):
            with h5py.File(idir+run.export_name(today)+"_scalars_2.h5",'r') as f:
                odata_2 = []
                for key in run.okeys_2:
                    odata_2.append(f['scalars'][key][:])
            
                odata_2 = list(np.array(odata_2).T)
            print("Adding to previous scalars_2 file.",flush=True)
        else:
            print("Making new scalars_2 file",flush=True)

    if (~np.isnan(NSc_3)):
        # Load scalar data to keep appending to
        odata_3 = []
        # Check if file exists
        if path.exists(idir+run.export_name(today)+"_scalars_3.h5"):
            with h5py.File(idir+run.export_name(today)+"_scalars_3.h5",'r') as f:
                odata_3 = []
                for key in run.okeys_3:
                    odata_3.append(f['scalars'][key][:])
            
                odata_3 = list(np.array(odata_3).T)
            print("Adding to previous scalars_3 file.",flush=True)
        else:
            print("Making new scalars_3 file",flush=True)
        
    # Load spec and flux averages to keep appending to
    # Check if file exists
    if path.exists(odir+'flux_spec_temp.npy'):
        [count,spec0_avg,spec0_var,spec1_avg,spec1_var,flux0_avg,flux0_var,flux1_avg,flux1_var,fluxA_avg,fluxA_var,thetauuu_hist,thetabbu_hist,thetabub_hist,thetaubb_hist,thetauuu_diff_hist,thetabbu_diff_hist,thetabub_diff_hist,thetaubb_diff_hist,phiu_hist,phib_hist,phiumb_hist] = np.load(odir+'flux_spec_temp.npy',allow_pickle=True)
        print("Continuing to average spectra and fluxes. Count = %s" % count,flush=True)
    else:
        # For averaging fluxes and spectra:
        count=2
        spec0_avg = np.real(run.u0*np.conj(run.u0))
        spec0_var = np.zeros(spec0_avg.shape)
        spec1_avg = np.real(run.u1*np.conj(run.u1))
        spec1_var = np.zeros(spec1_avg.shape)
        [U0,U1] = run.roll_fields(run.u0,run.u1)
        flux0_avg = run.Pi_0(U0,run.A0)
        flux0_var = np.zeros(flux0_avg.shape)
        flux1_avg = run.Pi_1(U0,U1,run.B0,run.A1,run.B1)
        flux1_var = np.zeros(flux1_avg.shape)
        fluxA_avg = run.Pi_A(U0,U1,run.B0,run.A1,run.B1,r3D)
        fluxA_var = np.zeros(fluxA_avg.shape)
        thetauuu_hist = np.zeros((nbins,run.N))
        thetabbu_hist = np.zeros((nbins,run.N))
        thetabub_hist = np.zeros((nbins,run.N))
        thetaubb_hist = np.zeros((nbins,run.N))
        thetauuu_diff_hist = np.zeros((nbins,run.N))
        thetabbu_diff_hist = np.zeros((nbins,run.N))
        thetabub_diff_hist = np.zeros((nbins,run.N))
        thetaubb_diff_hist = np.zeros((nbins,run.N))
        phiu_hist = np.zeros((nbins,run.N))
        phib_hist = np.zeros((nbins,run.N))
        phiumb_hist = np.zeros((nbins,run.N))
        print("NO FILE. Restarting the average spectra and fluxes from zero.",flush=True)
    
    print('Continuing from previous run %s' % (idir+run.export_name(today)),flush=True)

else:
    # Initialize scalar outputs:
    odata = []

    if (~np.isnan(NSc_2)):
        odata_2 = []

    if (~np.isnan(NSc_3)):
        odata_3 = []
    
    # For averaging fluxes and spectra:
    count=2
    spec0_avg = np.real(run.u0*np.conj(run.u0))
    spec0_var = np.zeros(spec0_avg.shape)
    spec1_avg = np.real(run.u1*np.conj(run.u1))
    spec1_var = np.zeros(spec1_avg.shape)
    [U0,U1] = run.roll_fields(run.u0,run.u1)
    flux0_avg = run.Pi_0(U0,run.A0)
    flux0_var = np.zeros(flux0_avg.shape)
    flux1_avg = run.Pi_1(U0,U1,run.B0,run.A1,run.B1)
    flux1_var = np.zeros(flux1_avg.shape)
    fluxA_avg = run.Pi_A(U0,U1,run.B0,run.A1,run.B1,r3D)
    fluxA_var = np.zeros(fluxA_avg.shape)
    thetauuu_hist = np.zeros((nbins,run.N))
    thetabbu_hist = np.zeros((nbins,run.N))
    thetabub_hist = np.zeros((nbins,run.N))
    thetaubb_hist = np.zeros((nbins,run.N))
    thetauuu_diff_hist = np.zeros((nbins,run.N))
    thetabbu_diff_hist = np.zeros((nbins,run.N))
    thetabub_diff_hist = np.zeros((nbins,run.N))
    thetaubb_diff_hist = np.zeros((nbins,run.N))
    phiu_hist = np.zeros((nbins,run.N))
    phib_hist = np.zeros((nbins,run.N))
    phiumb_hist = np.zeros((nbins,run.N))

    print('Starting run from zero',flush=True)

# Main loop
try:
    print('Starting time-stepping loop, tstep = %s' % (run.tstep),flush=True)
    start_time=time.time()
    sim_end = start_time + 60*60*H # run for H hours
    # Wait two seconds to not trigger state save:
    time.sleep(2)
    iter_start = run.tstep
    # Soft kill:
    size_lim=True # Becomes false if RUNNING.txt is deleted and kills the run
    
#    print('Setting time and tstep to zero!',flush=True)
#    run.tstep=1
#    run.t=0.0

    while (time.time() < sim_end)&(size_lim):        
        # Save the output every iter_state iterations during the run. All saved in one h5 file.
        # If restarting, outputs will simply add on to the existing file.
        
        #if (i>0)&(run.tstep % int(iter_state) == 0):
        #if run.tstep % int(iter_state) == 0:
        if ((time.time()-start_time) % ((60*60*H)/NS)) < 1: # Save NS times per run
            print('Saving state, tstep = %s, wall_time = %s' % (run.tstep,time.time()-start_time),flush=True)
            run.export_state(odir,today=today,overwrite=overwrite)
            # Save temporary spectra and fluxes
            np.save(odir+'flux_spec_temp.npy',np.array([
                count,
                spec0_avg,
                spec0_var,
                spec1_avg,
                spec1_var,
                flux0_avg,
                flux0_var,
                flux1_avg,
                flux1_var,
                fluxA_avg,
                fluxA_var,
                thetauuu_hist,
                thetabbu_hist,
                thetabub_hist,
                thetaubb_hist,
                thetauuu_diff_hist,
                thetabbu_diff_hist,
                thetabub_diff_hist,
                thetaubb_diff_hist,
                phiu_hist,
                phib_hist,
                phiumb_hist,
            ],dtype=object))
            overwrite=False
            time.sleep(1) # Avoids saving multiple times.

        if ((count*NSc_avg)%NSp)==0:
            print('Appending spectra, fluxes, and histograms, tstep = %s, wall_time = %s' % (run.tstep,time.time()-start_time),flush=True)

            # Initializing flux_spec file if it doesn't exist
            fname = odir+ run.export_name(today) +'_flux_spec.h5'
            
            new_file = not pathlib.Path(fname).exists()
            
            if (overwrite or new_file):
                new_write_spec=True
                with h5py.File(fname,'w') as f:
                    params = f.create_group('parameters')
                    for k, v in run.get_params().items():
                        params.create_dataset(k, data=np.array(v,dtype=np.float64))
                    scalars = f.create_group('flux_spec')
            else:
                new_write_spec=False
            
            with h5py.File(fname,'a') as f:
                scalars = f['flux_spec']
                if new_write_spec:
                    scalars.create_dataset('tstep',data=[run.tstep],shape=(1,1),maxshape=(None,1),chunks=True) 
                    scalars.create_dataset('t',data=[run.t],shape=(1,1),maxshape=(None,1),chunks=True) 
                    scalars.create_dataset('spec0_avg',data=[spec0_avg],shape=(1,N),maxshape=(None,N),chunks=True) 
                    scalars.create_dataset('spec0_var',data=[spec0_var],shape = (1,N),maxshape=(None,N),chunks=True) 
                    scalars.create_dataset('spec1_avg',data=[spec1_avg],shape=(1,N),maxshape=(None,N),chunks=True) 
                    scalars.create_dataset('spec1_var',data=[spec1_var],shape = (1,N),maxshape=(None,N),chunks=True)  
                    scalars.create_dataset('flux0_avg',data=[flux0_avg],shape=(1,N),maxshape=(None,N),chunks=True) 
                    scalars.create_dataset('flux0_var',data=[flux0_var],shape = (1,N),maxshape=(None,N),chunks=True)  
                    scalars.create_dataset('flux1_avg',data=[flux1_avg],shape=(1,N),maxshape=(None,N),chunks=True) 
                    scalars.create_dataset('flux1_var',data=[flux1_var],shape = (1,N),maxshape=(None,N),chunks=True)  
                    scalars.create_dataset('fluxA_avg',data=[fluxA_avg],shape=(1,N),maxshape=(None,N),chunks=True) 
                    scalars.create_dataset('fluxA_var',data=[fluxA_var],shape = (1,N),maxshape=(None,N),chunks=True) 
                    scalars.create_dataset('thetauuu_hist',data=[thetauuu_hist],shape = (1,nbins,run.N),maxshape=(None,nbins,run.N),chunks=True)    
                    scalars.create_dataset('thetabbu_hist',data=[thetabbu_hist],shape = (1,nbins,run.N),maxshape=(None,nbins,run.N),chunks=True)   
                    scalars.create_dataset('thetabub_hist',data=[thetabub_hist],shape = (1,nbins,run.N),maxshape=(None,nbins,run.N),chunks=True)   
                    scalars.create_dataset('thetaubb_hist',data=[thetaubb_hist],shape = (1,nbins,run.N),maxshape=(None,nbins,run.N),chunks=True)   
                    scalars.create_dataset('thetauuu_diff_hist',data=[thetauuu_diff_hist],shape = (1,nbins,run.N),maxshape=(None,nbins,run.N),chunks=True)    
                    scalars.create_dataset('thetabbu_diff_hist',data=[thetabbu_diff_hist],shape = (1,nbins,run.N),maxshape=(None,nbins,run.N),chunks=True)   
                    scalars.create_dataset('thetabub_diff_hist',data=[thetabub_diff_hist],shape = (1,nbins,run.N),maxshape=(None,nbins,run.N),chunks=True)   
                    scalars.create_dataset('thetaubb_diff_hist',data=[thetaubb_diff_hist],shape = (1,nbins,run.N),maxshape=(None,nbins,run.N),chunks=True)      
                    scalars.create_dataset('phiu_hist',data=[phiu_hist],shape = (1,nbins,run.N),maxshape=(None,nbins,run.N),chunks=True)         
                    scalars.create_dataset('phib_hist',data=[phib_hist],shape = (1,nbins,run.N),maxshape=(None,nbins,run.N),chunks=True)         
                    scalars.create_dataset('phiumb_hist',data=[phiumb_hist],shape = (1,nbins,run.N),maxshape=(None,nbins,run.N),chunks=True)  
                else:
                    scalars['tstep'].resize((scalars['tstep'].shape[0] + 1), axis = 0)
                    scalars['tstep'][-1:] = [run.tstep]
                    scalars['t'].resize((scalars['t'].shape[0] + 1), axis = 0)
                    scalars['t'][-1:] = [run.t]
                    scalars['spec0_avg'].resize((scalars['spec0_avg'].shape[0] + 1), axis = 0)
                    scalars['spec0_avg'][-1:] = [spec0_avg]
                    scalars['spec0_var'].resize((scalars['spec0_var'].shape[0] + 1), axis = 0)
                    scalars['spec0_var'][-1:] = [spec0_var]
                    scalars['spec1_avg'].resize((scalars['spec1_avg'].shape[0] + 1), axis = 0)
                    scalars['spec1_avg'][-1:] = [spec1_avg]
                    scalars['spec1_var'].resize((scalars['spec1_var'].shape[0] + 1), axis = 0)
                    scalars['spec1_var'][-1:] = [spec1_var]
                    scalars['flux0_avg'].resize((scalars['flux0_avg'].shape[0] + 1), axis = 0)
                    scalars['flux0_avg'][-1:] = [flux0_avg]
                    scalars['flux0_var'].resize((scalars['flux0_var'].shape[0] + 1), axis = 0)
                    scalars['flux0_var'][-1:] = [flux0_var]
                    scalars['flux1_avg'].resize((scalars['flux1_avg'].shape[0] + 1), axis = 0)
                    scalars['flux1_avg'][-1:] = [flux1_avg]
                    scalars['flux1_var'].resize((scalars['flux1_var'].shape[0] + 1), axis = 0)
                    scalars['flux1_var'][-1:] = [flux1_var]
                    scalars['fluxA_avg'].resize((scalars['fluxA_avg'].shape[0] + 1), axis = 0)
                    scalars['fluxA_avg'][-1:] = [fluxA_avg]
                    scalars['fluxA_var'].resize((scalars['fluxA_var'].shape[0] + 1), axis = 0)
                    scalars['fluxA_var'][-1:] = [fluxA_var]
                    scalars['thetauuu_hist'].resize((scalars['thetauuu_hist'].shape[0] + 1), axis = 0)
                    scalars['thetauuu_hist'][-1:] = [thetauuu_hist]
                    scalars['thetabbu_hist'].resize((scalars['thetabbu_hist'].shape[0] + 1), axis = 0)
                    scalars['thetabbu_hist'][-1:] = [thetabbu_hist]
                    scalars['thetabub_hist'].resize((scalars['thetabub_hist'].shape[0] + 1), axis = 0)
                    scalars['thetabub_hist'][-1:] = [thetabub_hist]
                    scalars['thetaubb_hist'].resize((scalars['thetaubb_hist'].shape[0] + 1), axis = 0)
                    scalars['thetaubb_hist'][-1:] = [thetaubb_hist]
                    scalars['thetauuu_diff_hist'].resize((scalars['thetauuu_diff_hist'].shape[0] + 1), axis = 0)
                    scalars['thetauuu_diff_hist'][-1:] = [thetauuu_diff_hist]
                    scalars['thetabbu_diff_hist'].resize((scalars['thetabbu_diff_hist'].shape[0] + 1), axis = 0)
                    scalars['thetabbu_diff_hist'][-1:] = [thetabbu_diff_hist]
                    scalars['thetabub_diff_hist'].resize((scalars['thetabub_diff_hist'].shape[0] + 1), axis = 0)
                    scalars['thetabub_diff_hist'][-1:] = [thetabub_diff_hist]
                    scalars['thetaubb_diff_hist'].resize((scalars['thetaubb_diff_hist'].shape[0] + 1), axis = 0)
                    scalars['thetaubb_diff_hist'][-1:] = [thetaubb_diff_hist]
                    scalars['phiu_hist'].resize((scalars['phiu_hist'].shape[0] + 1), axis = 0)
                    scalars['phiu_hist'][-1:] = [phiu_hist]
                    scalars['phib_hist'].resize((scalars['phib_hist'].shape[0] + 1), axis = 0)
                    scalars['phib_hist'][-1:] = [phib_hist]
                    scalars['phiumb_hist'].resize((scalars['phiumb_hist'].shape[0] + 1), axis = 0)
                    scalars['phiumb_hist'][-1:] = [phiumb_hist]

            # Resets averages and variance after saving to file.
            spec0_avg = np.real(run.u0*np.conj(run.u0))
            spec0_var = np.zeros(spec0_avg.shape)
            spec1_avg = np.real(run.u1*np.conj(run.u1))
            spec1_var = np.zeros(spec1_avg.shape)
            [U0,U1] = run.roll_fields(run.u0,run.u1)
            flux0_avg = run.Pi_0(U0,run.A0)
            flux0_var = np.zeros(flux0_avg.shape)
            flux1_avg = run.Pi_1(U0,U1,run.B0,run.A1,run.B1)
            flux1_var = np.zeros(flux1_avg.shape)
            fluxA_avg = run.Pi_A(U0,U1,run.B0,run.A1,run.B1,r3D)
            fluxA_var = np.zeros(fluxA_avg.shape)
            thetauuu_hist = np.zeros((nbins,run.N))
            thetabbu_hist = np.zeros((nbins,run.N))
            thetabub_hist = np.zeros((nbins,run.N))
            thetaubb_hist = np.zeros((nbins,run.N))
            thetauuu_diff_hist = np.zeros((nbins,run.N))
            thetabbu_diff_hist = np.zeros((nbins,run.N))
            thetabub_diff_hist = np.zeros((nbins,run.N))
            thetaubb_diff_hist = np.zeros((nbins,run.N))
            phiu_hist = np.zeros((nbins,run.N))
            phib_hist = np.zeros((nbins,run.N))
            phiumb_hist = np.zeros((nbins,run.N))
            count=2
        
        if (run.tstep%NSc_avg)==0:
            # Update average profiles
            spec0_avg_new = spec0_avg + (np.real(run.u0*np.conj(run.u0))-spec0_avg)/count
            spec0_var = spec0_var + ((np.real(run.u0*np.conj(run.u0))-spec0_avg)*(np.real(run.u0*np.conj(run.u0))-spec0_avg_new) - spec0_var)/count
            spec0_avg = spec0_avg_new
            spec1_avg_new = spec1_avg + (np.real(run.u1*np.conj(run.u1))-spec1_avg)/count
            spec1_var = spec1_var + ((np.real(run.u1*np.conj(run.u1))-spec1_avg)*(np.real(run.u1*np.conj(run.u1))-spec1_avg_new) - spec1_var)/count
            spec1_avg = spec1_avg_new
            [U0,U1] = run.roll_fields(run.u0,run.u1)
            flux0_avg_new = flux0_avg + (run.Pi_0(U0,run.A0)-flux0_avg)/count
            flux0_var = flux0_var + ((run.Pi_0(U0,run.A0)-flux0_avg)*(run.Pi_0(U0,run.A0)-flux0_avg_new) - flux0_var)/count
            flux0_avg = flux0_avg_new
            flux1_avg_new = flux1_avg + (run.Pi_1(U0,U1,run.B0,run.A1,run.B1)-flux1_avg)/count
            flux1_var = flux1_var + ((run.Pi_1(U0,U1,run.B0,run.A1,run.B1)-flux1_avg)*(run.Pi_1(U0,U1,run.B0,run.A1,run.B1)-flux1_avg_new) - flux1_var)/count
            flux1_avg = flux1_avg_new
            fluxA_avg_new = fluxA_avg + (run.Pi_A(U0,U1,run.B0,run.A1,run.B1,r3D)-fluxA_avg)/count
            fluxA_var = fluxA_var + ((run.Pi_A(U0,U1,run.B0,run.A1,run.B1,r3D)-fluxA_avg)*(run.Pi_A(U0,U1,run.B0,run.A1,run.B1,r3D)-fluxA_avg_new) - fluxA_var)/count
            fluxA_avg = fluxA_avg_new
            # PDFS:
            # Calculate the triad phases
            [U0,U1] = run.roll_fields(run.u0,run.u1)
            phi0k = np.angle(U0[2])
            phi0kp1 = np.angle(U0[3])
            phi0kp2 = np.angle(U0[4])
            phi1k = np.angle(U1[2])
            phi1kp1 = np.angle(U1[3])
            phi1kp2 = np.angle(U1[4])
            
            thetauuu = (phi0kp2-phi0kp1-phi0k)
            thetabbu = (phi1kp2-phi1kp1-phi0k)
            thetabub = (phi1kp2-phi0kp1-phi1k)
            thetaubb = (phi0kp2-phi1kp1-phi1k)
            thetauuu_diff = thetauuu - np.roll(thetauuu,1)
            thetabbu_diff = thetabbu - np.roll(thetabbu,1)
            thetabub_diff = thetabub - np.roll(thetabub,1)
            thetaubb_diff = thetaubb - np.roll(thetaubb,1)
            phiumb = (phi0k-phi1k)
            
            thetauuu = thetauuu -2*np.pi*np.round(thetauuu/np.pi/2) # From [-pi,pi]
            thetabbu = thetabbu -2*np.pi*np.round(thetabbu/np.pi/2) # From [-pi,pi]
            thetabub = thetabub -2*np.pi*np.round(thetabub/np.pi/2) # From [-pi,pi]
            thetaubb = thetaubb -2*np.pi*np.round(thetaubb/np.pi/2) # From [-pi,pi]
            thetauuu_diff = thetauuu_diff -2*np.pi*np.round(thetauuu_diff/np.pi/2) # From [-pi,pi]
            thetabbu_diff = thetabbu_diff -2*np.pi*np.round(thetabbu_diff/np.pi/2) # From [-pi,pi]
            thetabub_diff = thetabub_diff -2*np.pi*np.round(thetabub_diff/np.pi/2) # From [-pi,pi]
            thetaubb_diff = thetaubb_diff -2*np.pi*np.round(thetaubb_diff/np.pi/2) # From [-pi,pi]
            phiumb = phiumb -2*np.pi*np.round(phiumb/np.pi/2) # From [-pi,pi]
            for jj in range(N):
                induuu = np.argmin(np.abs(thetauuu[jj]-bin_edges_centered))
                thetauuu_hist[induuu,jj]+=1
                indbbu = np.argmin(np.abs(thetabbu[jj]-bin_edges_centered))
                thetabbu_hist[indbbu,jj]+=1
                indbub = np.argmin(np.abs(thetabub[jj]-bin_edges_centered))
                thetabub_hist[indbub,jj]+=1
                indubb = np.argmin(np.abs(thetaubb[jj]-bin_edges_centered))
                thetaubb_hist[indubb,jj]+=1                
                induuu_diff = np.argmin(np.abs(thetauuu_diff[jj]-bin_edges_centered))
                thetauuu_diff_hist[induuu_diff,jj]+=1
                indbbu_diff = np.argmin(np.abs(thetabbu_diff[jj]-bin_edges_centered))
                thetabbu_diff_hist[indbbu_diff,jj]+=1
                indbub_diff = np.argmin(np.abs(thetabub_diff[jj]-bin_edges_centered))
                thetabub_diff_hist[indbub_diff,jj]+=1
                indubb_diff = np.argmin(np.abs(thetaubb_diff[jj]-bin_edges_centered))
                thetaubb_diff_hist[indubb_diff,jj]+=1
                indu = np.argmin(np.abs(phi0k[jj]-bin_edges_centered))
                phiu_hist[indu,jj]+=1
                indb = np.argmin(np.abs(phi1k[jj]-bin_edges_centered))
                phib_hist[indb,jj]+=1
                indumb = np.argmin(np.abs(phiumb[jj]-bin_edges_centered))
                phiumb_hist[indumb,jj]+=1
                
            count+=1
            
        if (run.tstep%NSc)==0:
            # Add to the output scalars:
            odata.append(run.get_scalars())            

        if ((run.tstep%NSc_2)==0)&(~np.isnan(NSc_2)):
            # Add to the other output scalars:
            odata_2.append(run.get_scalars_2()) 

        if ((run.tstep%NSc_3)==0)&(~np.isnan(NSc_3)):
            # Add to the other output scalars:
            odata_3.append(run.get_scalars_3()) 
            
        # Check size every 15 minutes (otherwise it reduces time-step):
        if (((time.time()-start_time) % (60*15)) < 1):
            # Initializing flux_spec file if it doesn't exist
            run_file = not pathlib.Path('RUNNING.txt').exists()
            if run_file:
                print("RUNNING.txt removed, stopping.",flush=True)
                size_lim = False
            
            # Check size of files (otherwise there is an issue and the run crashes).
            odsize = np.array(odata).nbytes/1024**2
            if odsize>512: # Memory limit is 512 MB
                print("Odata reached it's maximum size of 512 MB, saving overflow and restarting odata.",flush=True)
                # Determine what number overflow (if any).
                test_dirs = glob.glob(idir+'shell_data*_scalars_overflow*.h5')
                if len(test_dirs)>0:
                    nover = len(test_dirs)+1
                else:
                    nover = 1
                run.export_scalars(odir,odata,today=today,overwrite=overwrite,name='overflow_'+str(nover)) #If loading from previous run, make sure 'overwrite' is 0 (false), so it updates the same file.
                odata = []
            
            if (~np.isnan(NSc_2)):
                odsize = np.array(odata_2).nbytes/1024**2
                if odsize>512: # Memory limit is 512 MB
                    print("Odata_2 reached it's maximum size of 512 MB, saving overflow and restarting odata.",flush=True)
                    # Determine what number overflow (if any).
                    test_dirs = glob.glob(idir+'shell_data*_scalars_2_overflow*.h5')
                    if len(test_dirs)>0:
                        nover = len(test_dirs)+1
                    else:
                        nover = 1
                    run.export_scalars_2(odir,odata_2,today=today,overwrite=overwrite,name='overflow_'+str(nover)) #If loading from previous run, make sure 'overwrite' is 0 (false), so it updates the same file.
                    odata_2 = []
            if (~np.isnan(NSc_3)):
                odsize = np.array(odata_3).nbytes/1024**2
                if odsize>512: # Memory limit is 512 MB
                    print("Odata_3 reached it's maximum size of 512 MB, saving overflow and restarting odata.",flush=True)
                    # Determine what number overflow (if any).
                    test_dirs = glob.glob(idir+'shell_data*_scalars_3_overflow*.h5')
                    if len(test_dirs)>0:
                        nover = len(test_dirs)+1
                    else:
                        nover = 1
                    run.export_scalars_3(odir,odata_3,today=today,overwrite=overwrite,name='overflow_'+str(nover)) #If loading from previous run, make sure 'overwrite' is 0 (false), so it updates the same file.
                    odata_3 = []

                
            time.sleep(1) # Avoids checking multiple times.

        # Take a time-step in the model:
        # run.step()
        if ((eps1 == 0.0)&(u1_init== 0.0)):
            run.step_HD()
        else:
            run.step()

    iter_end = run.tstep
    end_time=time.time()
    print('Finished time-stepping loop. Total real time: %.4f, iterations per second: %.4f.' % (end_time-start_time,(iter_end-iter_start)/(end_time-start_time)),flush=True)
except:
    raise
finally: 
    # Saving output
    print('Saving final state and scalars, tstep = %s' % (run.tstep),flush=True)
    run.export_state(odir,today=today,overwrite=overwrite)
    # Save scalars at the end of the run:

    run.export_scalars(odir,odata,today=today,overwrite=overwrite) #If loading from previous run, make sure 'overwrite' is 0 (false), so it updates the same file.

    if (~np.isnan(NSc_2)):
        run.export_scalars_2(odir,odata_2,today=today,overwrite=overwrite) #If loading from previous run, make sure 'overwrite' is 0 (false), so it updates the same file.
        
    if (~np.isnan(NSc_3)):
        run.export_scalars_3(odir,odata_3,today=today,overwrite=overwrite) #If loading from previous run, make sure 'overwrite' is 0 (false), so it updates the same file.
    
    # Save temporary spectra and fluxes
    np.save(odir+'flux_spec_temp.npy',np.array([
        count,
        spec0_avg,
        spec0_var,
        spec1_avg,
        spec1_var,
        flux0_avg,
        flux0_var,
        flux1_avg,
        flux1_var,
        fluxA_avg,
        fluxA_var,
        thetauuu_hist,
        thetabbu_hist,
        thetabub_hist,
        thetaubb_hist,
        thetauuu_diff_hist,
        thetabbu_diff_hist,
        thetabub_diff_hist,
        thetaubb_diff_hist,
        phiu_hist,
        phib_hist,
        phiumb_hist,
    ],dtype=object))
    
    if rank==0:
        run_file = pathlib.Path('RUNNING.txt').exists()
        if run_file:
            os.remove('RUNNING.txt')

    print('Finished saving. Exiting... \n \n',flush=True)
