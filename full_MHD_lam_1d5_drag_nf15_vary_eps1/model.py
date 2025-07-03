"""
shell superclass
"""
import numpy as np
import tqdm
import h5py
from os import path
import random
from datetime import date

class shell():
    
    def __init__(self,nf,N,dt,lam,A0,B0,A1,B1,eps0,eps1,Q,nu,nu_l,u0_init = 1e-4, u1_init = 0,alpha=0,beta=2):
        ## Input parameters to be communicated to other functions:        
        self.nf = int(nf)
        self.N = int(N)
        self.lam = lam
        self.A0 = A0 # 2D parameters A0 = [a0,b0,c0] (following a,b,c in Sabra)
        self.B0 = B0 # u1 NL term in u0 equation. parameters B0 = [a0,b0,c0] (following a,b,c in Sabra)
        self.A1 = A1 # 3D parameters A1 = [a1,b1,c1]
        self.B1 = B1 # 3D parameters B1 = [a1,b1,c1]
        self.eps0 = eps0
        self.eps1 = eps1
        self.Q = Q
        self.dt = dt
        
        ################
        ### Building ###
        ################
        self.k0 = self.lam**(-self.nf)
        self.n = np.arange(self.N,dtype=float)
        self.k = self.k0*self.lam**self.n
        self.km1 = np.roll(self.k,1)
        self.kp1 = np.roll(self.k,-1)

        if self.eps0>0:
            self.nu = nu*((2**(-4)*2**24)/(self.k0*self.lam**self.N))**(4/3)*(self.eps0)**(1/3) # Viscosity (k^2). Chosen so that k_diss is k_N (based on original values found in Cocciaglia et al (arXiv) 2023).
        else:
            self.nu = nu*((2**(-4)*2**24)/(self.k0*self.lam**self.N))**(4/3)*(self.eps1)**(1/3)
        self.nu_l = nu_l*(self.eps0)**(1/3)*((self.k0)/(self.lam**(-10)))**(2/3) # Drag. Chosen so that k_{L,diss} is at k0
        # Drag only acts on first two modes.
        filt = np.zeros(self.k.shape)
        filt[:2] = 1.0
        self.L0 = -(self.nu*self.k**2 + self.nu_l*filt) # Linear operator
        self.L1 = -(self.nu*(self.k**2+self.Q**2)+ self.nu_l*filt) # Linear operator
        
        ####################
        ## INITIAL fields ##
        ####################
        # For random numbers:
        self.rng = np.random.default_rng(12345)
        
        # Start with random u
        phi = self.rng.uniform(low=-np.pi/2,high=np.pi/2,size=(self.n.shape))
        self.u0 = np.sqrt(u0_init)*(np.cos(phi)+1j*np.sin(phi))*self.k**(alpha/2) # So that the u0 _energy_ has a 'alpha' spectral slope.
        # Make KE(n>=nf) have a -1 slope.
        # self.u0[self.n>=self.nf] = np.sqrt(u0_init)*(np.cos(phi[self.n>=self.nf])+1j*np.sin(phi[self.n>=self.nf]))*self.k[self.n>=self.nf]**(-1/2)
        
        phi = self.rng.uniform(low=-np.pi/2,high=np.pi/2,size=(self.n.shape))
        self.u1 = np.sqrt(u1_init)*(np.cos(phi)+1j*np.sin(phi))*self.k**(beta/2) # So that the u1 _energy_ has a 'beta' spectral slope.
        # Make ME have a -1 slope when KE = ME.
        #cond = np.abs(self.u1)>=np.abs(self.u0)
        #self.u1[cond] = self.u0[cond]
        ## 
        self.f0 = np.zeros(self.n.shape,dtype=complex)
        self.f1 = np.zeros(self.n.shape,dtype=complex)
        self.tstep = 1
        self.t = 0.0
        
        ## Output keys:
        self.okeys = ['tstep','t','en_u0','en_u1','diss_u0','diss_u1','hdiss_u0','hdiss_u1','h_u0','h_u1','diss_A','hdiss_A','en_u0_nf','en_u1_nf'] # These will be output less frequently, since their averages don't require very long time-averages.
        self.okeys_2 = ['tstep','t','flux0','flux1','fluxA','thetauuu_IC_m2','thetauuu_IC_m1','thetauuu_IC','thetauuu_IC_p1','thetauuu_IC_p2','thetabbu_IC_m2','thetabbu_IC_m1','thetabbu_IC','thetabbu_IC_p1','thetabbu_IC_p2','thetabub_IC_m2','thetabub_IC_m1','thetabub_IC','thetabub_IC_p1','thetabub_IC_p2','thetaubb_IC_m2','thetaubb_IC_m1','thetaubb_IC','thetaubb_IC_p1','thetaubb_IC_p2','thetauuu_FC_m2','thetauuu_FC_m1','thetauuu_FC','thetauuu_FC_p1','thetauuu_FC_p2','thetabbu_FC_m2','thetabbu_FC_m1','thetabbu_FC','thetabbu_FC_p1','thetabbu_FC_p2','thetabub_FC_m2','thetabub_FC_m1','thetabub_FC','thetabub_FC_p1','thetabub_FC_p2','thetaubb_FC_m2','thetaubb_FC_m1','thetaubb_FC','thetaubb_FC_p1','thetaubb_FC_p2','delta_IC_m2','delta_IC_m1','delta_IC','delta_IC_p1','delta_IC_p2','delta_FC_m2','delta_FC_m1','delta_FC','delta_FC_p1','delta_FC_p2'] # These could be output more frequently for longer time averages.
        
        self.okeys_3 = ['tstep','t','theta_out[N/4]','coeff_self[N/4]','theta_rest[N/4]','theta_out[N/3]','coeff_self[N/3]','theta_rest[N/3]','theta_out[N/2]','coeff_self[N/2]','theta_rest[N/2]','theta_out[2N/3]','coeff_self[2N/3]','theta_rest[2N/3]','theta_out[3N/4]','coeff_self[3N/4]','theta_rest[3N/4]']
    
    
    #################
    ### Functions ###
    #################      
    def roll_fields(self,u0,u1):
        # Call roll externally
        u0m2 = np.roll(u0,2)
        u0m1 = np.roll(u0,1)
        u0p1 = np.roll(u0,-1)
        u0p2 = np.roll(u0,-2)
        u1m2 = np.roll(u1,2)
        u1m1 = np.roll(u1,1)
        u1p1 = np.roll(u1,-1)
        u1p2 = np.roll(u1,-2)

        # Boundary conditions:
        u0m2[:2] = 0.0
        u0m1[:1] = 0.0
        u0p1[-1:] = 0.0
        u0p2[-2:] = 0.0
        u1m2[:2] = 0.0
        u1m1[:1] = 0.0
        u1p1[-1:] = 0.0
        u1p2[-2:] = 0.0
        
        return [[u0m2,u0m1,u0,u0p1,u0p2],[u1m2,u1m1,u1,u1p1,u1p2]]

    def roll_fields_HD(self,u0):
        # Call roll externally
        u0m2 = np.roll(u0,2)
        u0m1 = np.roll(u0,1)
        u0p1 = np.roll(u0,-1)
        u0p2 = np.roll(u0,-2)

        # Boundary conditions:
        u0m2[:2] = 0.0
        u0m1[:1] = 0.0
        u0p1[-1:] = 0.0
        u0p2[-2:] = 0.0
        
        return [u0m2,u0m1,u0,u0p1,u0p2]
    
    def NL_1(self,U,A):
        """ U = [um2,um1,u,up1,up2]"""
        return 1j*(A[0]*self.kp1*U[4]*np.conj(U[3]) + A[1]*self.k*U[3]*np.conj(U[1]) - A[2]*self.km1*U[1]*U[0])
    
    def NL_2(self,X,Y,A):
        """ X = [Xm2,Xm1,X,Xp1,Xp2]"""
        return 1j*(A[0]*self.kp1*Y[4]*np.conj(X[3]) + A[1]*self.k*Y[3]*np.conj(X[1]) - A[2]*self.km1*Y[1]*X[0])

    def Pi_0(self,U,A):
        return self.k*np.imag(A[0]*self.lam*np.conj(U[2])*np.conj(U[3])*U[4] + (A[0]+A[1])*np.conj(U[1])*np.conj(U[2])*U[3])
    
    def Pi_1(self,X,Y,B0,A1,B1):
        
        Pi1 = self.k*np.imag(B0[0]*self.lam*np.conj(X[2])*np.conj(Y[3])*Y[4] + (B0[0]+A1[1])*np.conj(X[1])*np.conj(Y[2])*Y[3]) 
        Pi2 = self.k*np.imag(A1[0]*self.lam*np.conj(Y[2])*np.conj(X[3])*Y[4] + (A1[0]+B0[1])*np.conj(Y[1])*np.conj(X[2])*Y[3])
        Pi3 = self.k*np.imag(B1[0]*self.lam*np.conj(Y[2])*np.conj(Y[3])*X[4] + (B1[0]+B1[1])*np.conj(Y[1])*np.conj(Y[2])*X[3])
        return Pi1+Pi2+Pi3

    def Pi_A(self,X,Y,B0,A1,B1,exp):
        
        Pi1 = -2*self.lam**(2*exp)*B1[2]*np.imag(self.k*(self.k/self.lam)**(exp)*np.conj(Y[1])*np.conj(X[2])*Y[3] + self.k**(exp)*self.k*self.lam*np.conj(Y[2])*np.conj(X[3])*Y[4])
        Pi2 = -2*self.lam**(exp)*A1[2]*np.imag(self.k**(exp+1)*np.conj(Y[2])*np.conj(X[1])*Y[3])
        Pi3 = -2*self.lam**(exp)*B1[1]*np.imag(self.k*self.lam*self.k**(exp)*np.conj(Y[2])*np.conj(Y[3])*X[4])
        
        return Pi1+Pi2+Pi3
    
    def energy(self,u,exp):
        """
        Input: u (field), exp (float)
        
        Outputs sum_n [k_n**(exp) u_n * conj(u_n)]
        """
        return np.sum(self.k**exp*np.real(u*np.conj(u)))
    
    def energy_alt(self,u,exp):
        """
        Input: u (field), exp (float)
        
        Outputs sum_n [k_n**(exp) (-1)^n u_n * conj(u_n)]
        """
        temp = self.k**exp*np.real(u*np.conj(u))
        return np.sum(temp[::2])-np.sum(temp[1::2])


    ####################
    # Take a time step #
    ####################
    def step(self):     
        """
        Take a time-step. Dynamical inputs needed: u0, u1. Returns nothing, just updates [u0,u1].

        """
        ########
        #### RK4 
        # Intermediate steps    
        ##### k1: f(t,w)
        if np.any(self.u0[int(self.nf):int(self.nf+2)]==0):
            self.f0[int(self.nf):int(self.nf+2)] = 0.0
        else:
            self.f0[int(self.nf):int(self.nf+2)] = self.eps0/np.conj(2*self.u0[int(self.nf):int(self.nf+2)])
        if np.any(self.u1[int(self.nf):int(self.nf+2)]==0):
            self.f1[int(self.nf):int(self.nf+2)] = 0.0
        else:
            self.f1[int(self.nf):int(self.nf+2)] = self.eps1/np.conj(2*self.u1[int(self.nf):int(self.nf+2)])
        [U0,U1] = self.roll_fields(self.u0,self.u1)
        k1 = ( self.f0 + self.NL_1(U0,self.A0) + self.NL_1(U1,self.B0)) 
        p1 = ( self.f1 + self.NL_2(U0,U1,self.A1) + self.NL_2(U1,U0,self.B1)) 

        ##### k2: f(t+h/2, w + h*k1/2)
        wtemp0 = np.exp(self.L0*(self.dt/2)) * (self.u0 + k1*self.dt/2)
        wtemp1 = np.exp(self.L1*(self.dt/2)) * (self.u1 + p1*self.dt/2)
        if np.any(wtemp0[int(self.nf):int(self.nf+2)]==0):
            self.f0[int(self.nf):int(self.nf+2)] = 0.0
        else:
            self.f0[int(self.nf):int(self.nf+2)] = self.eps0/np.conj(2*wtemp0[int(self.nf):int(self.nf+2)])
        if np.any(wtemp1[int(self.nf):int(self.nf+2)]==0):
            self.f1[int(self.nf):int(self.nf+2)] = 0.0
        else:
            self.f1[int(self.nf):int(self.nf+2)] = self.eps1/np.conj(2*wtemp1[int(self.nf):int(self.nf+2)])
        [U0,U1] = self.roll_fields(wtemp0,wtemp1)
        k2 = np.exp(-self.L0*(self.dt/2)) * ( self.f0 + self.NL_1(U0,self.A0) + self.NL_1(U1,self.B0) ) 
        p2 = np.exp(-self.L1*(self.dt/2)) * ( self.f1 + self.NL_2(U0,U1,self.A1) + self.NL_2(U1,U0,self.B1) ) 

        ##### k3: f(t+h/2, w + h*k2/2)
        wtemp0 = np.exp(self.L0*(self.dt/2)) * (self.u0 + k2*self.dt/2)
        wtemp1 = np.exp(self.L1*(self.dt/2)) * (self.u1 + p2*self.dt/2)
        if np.any(wtemp0[int(self.nf):int(self.nf+2)]==0):
            self.f0[int(self.nf):int(self.nf+2)] = 0.0
        else:
            self.f0[int(self.nf):int(self.nf+2)] = self.eps0/np.conj(2*wtemp0[int(self.nf):int(self.nf+2)])
        if np.any(wtemp1[int(self.nf):int(self.nf+2)]==0):
            self.f1[int(self.nf):int(self.nf+2)] = 0.0
        else:
            self.f1[int(self.nf):int(self.nf+2)] = self.eps1/np.conj(2*wtemp1[int(self.nf):int(self.nf+2)])
        [U0,U1] = self.roll_fields(wtemp0,wtemp1)
        k3 = np.exp(-self.L0*(self.dt/2)) * ( self.f0 + self.NL_1(U0, self.A0) + self.NL_1(U1, self.B0) ) 
        p3 = np.exp(-self.L1*(self.dt/2)) * ( self.f1 + self.NL_2(U0,U1,self.A1) + self.NL_2(U1,U0,self.B1) ) 

        ##### k4: f(t+h, w + h * k3)
        wtemp0 = np.exp(self.L0*(self.dt)) * (self.u0 + k3*self.dt) 
        wtemp1 = np.exp(self.L1*(self.dt)) * (self.u1 + p3*self.dt) 
        if np.any(wtemp0[int(self.nf):int(self.nf+2)]==0):
            self.f0[int(self.nf):int(self.nf+2)] = 0.0
        else:
            self.f0[int(self.nf):int(self.nf+2)] = self.eps0/np.conj(2*wtemp0[int(self.nf):int(self.nf+2)])
        if np.any(wtemp1[int(self.nf):int(self.nf+2)]==0):
            self.f1[int(self.nf):int(self.nf+2)] = 0.0
        else:
            self.f1[int(self.nf):int(self.nf+2)] = self.eps1/np.conj(2*wtemp1[int(self.nf):int(self.nf+2)])
        [U0,U1] = self.roll_fields(wtemp0,wtemp1)
        k4 = np.exp(-self.L0*(self.dt)) * ( self.f0 + self.NL_1( U0, self.A0) + self.NL_1( U1, self.B0))
        p4 = np.exp(-self.L1*(self.dt)) * ( self.f1 + self.NL_2( U0, U1, self.A1) + self.NL_2( U1,U0, self.B1))

        # Update final step
        self.u0 = np.exp(self.L0*self.dt)*self.u0 + np.exp(self.L0*self.dt) * (self.dt/6) * ( k1 + 2*k2 + 2*k3 + k4)
        self.u1 = np.exp(self.L1*self.dt)*self.u1 + np.exp(self.L1*self.dt) * (self.dt/6) * ( p1 + 2*p2 + 2*p3 + p4)

        # Update time
        self.t += self.dt
        self.tstep += 1
        
        return

    def step_HD(self):     
        """
        Take a time-step. Dynamical inputs needed: u0. Returns nothing, just updates u0.

        """
        ########
        #### RK4 
        # Intermediate steps    
        ##### k1: f(t,w)
        if np.any(self.u0[int(self.nf):int(self.nf+2)]==0):
            self.f0[int(self.nf):int(self.nf+2)] = 0.0
        else:
            self.f0[int(self.nf):int(self.nf+2)] = self.eps0/np.conj(2*self.u0[int(self.nf):int(self.nf+2)])
        U0 = self.roll_fields_HD(self.u0)
        k1 = self.f0 + self.NL_1(U0,self.A0) 

        ##### k2: f(t+h/2, w + h*k1/2)
        wtemp0 = np.exp(self.L0*(self.dt/2)) * (self.u0 + k1*self.dt/2)
        if np.any(wtemp0[int(self.nf):int(self.nf+2)]==0):
            self.f0[int(self.nf):int(self.nf+2)] = 0.0
        else:
            self.f0[int(self.nf):int(self.nf+2)] = self.eps0/np.conj(2*wtemp0[int(self.nf):int(self.nf+2)])
        U0 = self.roll_fields_HD(wtemp0)
        k2 = np.exp(-self.L0*(self.dt/2)) * ( self.f0 + self.NL_1(U0,self.A0) ) 

        ##### k3: f(t+h/2, w + h*k2/2)
        wtemp0 = np.exp(self.L0*(self.dt/2)) * (self.u0 + k2*self.dt/2)
        if np.any(wtemp0[int(self.nf):int(self.nf+2)]==0):
            self.f0[int(self.nf):int(self.nf+2)] = 0.0
        else:
            self.f0[int(self.nf):int(self.nf+2)] = self.eps0/np.conj(2*wtemp0[int(self.nf):int(self.nf+2)])
        U0 = self.roll_fields_HD(wtemp0)
        k3 = np.exp(-self.L0*(self.dt/2)) * ( self.f0 + self.NL_1(U0, self.A0) )

        ##### k4: f(t+h, w + h * k3)
        wtemp0 = np.exp(self.L0*(self.dt)) * (self.u0 + k3*self.dt) 
        if np.any(wtemp0[int(self.nf):int(self.nf+2)]==0):
            self.f0[int(self.nf):int(self.nf+2)] = 0.0
        else:
            self.f0[int(self.nf):int(self.nf+2)] = self.eps0/np.conj(2*wtemp0[int(self.nf):int(self.nf+2)])
        U0 = self.roll_fields_HD(wtemp0)
        k4 = np.exp(-self.L0*(self.dt)) * ( self.f0 + self.NL_1( U0, self.A0) )

        # Update final step
        self.u0 = np.exp(self.L0*self.dt)*self.u0 + np.exp(self.L0*self.dt) * (self.dt/6) * ( k1 + 2*k2 + 2*k3 + k4)

        # Update time
        self.t += self.dt
        self.tstep += 1
        
        return

    def step_phase_only(self):     
        """
        Take a time-step, but only advance the phases and keep the magnitudes the same. Because viscosity doesn't play a role, we remove the linear operator. 
        Dynamical inputs needed: u0, u1. Returns nothing, just updates [u0,u1].
        """
        ## The magnitudes are fixed, so we will define rho0 and rho1 and keep them fixed.
        rho0 = np.abs(self.u0)
        rho1 = np.abs(self.u1)
        
        # ## Forcing won't change over the course of the RK loop, since it's only a function of rho.
        # if np.any(self.u0[int(self.nf):int(self.nf+2)]==0):
        #     self.f0[int(self.nf):int(self.nf+2)] = 0.0
        # else:
        #     self.f0[int(self.nf):int(self.nf+2)] = self.eps0/np.conj(2*rho0[int(self.nf):int(self.nf+2)]**2)
        # if np.any(self.u1[int(self.nf):int(self.nf+2)]==0):
        #     self.f1[int(self.nf):int(self.nf+2)] = 0.0
        # else:
        #     self.f1[int(self.nf):int(self.nf+2)] = self.eps1/np.conj(2*rho1[int(self.nf):int(self.nf+2)]**2)
        # # We will evolve the phases only now.
        
        ########
        #### RK4 
        # Intermediate steps    
        ##### k1: f(t,w)
        # Only updating the phase, so we need to apply Imag[exp(-1j*phi)*f / rho].
        phi0 = np.angle(self.u0)
        phi1 = np.angle(self.u1)
        [U0,U1] = self.roll_fields(rho0*np.exp(1j*phi0),rho1*np.exp(1j*phi1))
        k1 = np.zeros(self.u0.shape)
        k1[rho0>0] = np.imag( np.exp(-1j*phi0[rho0>0]) * ( self.NL_1(U0,self.A0)[rho0>0] + self.NL_1(U1,self.B0)[rho0>0])  / rho0[rho0>0]) #+ self.f0
        p1 = np.zeros(self.u1.shape)
        p1[rho1>0] = np.imag( np.exp(-1j*phi1[rho1>0]) * ( self.NL_2(U0,U1,self.A1)[rho1>0] + self.NL_2(U1,U0,self.B1)[rho1>0])  / rho1[rho1>0]) #+ self.f1
        
        ##### k2: f(t+h/2, w + h*k1/2)
        phi0 = (np.angle(self.u0) + k1*self.dt/2)
        phi1 = (np.angle(self.u1) + p1*self.dt/2)
        [U0,U1] = self.roll_fields(rho0*np.exp(1j*phi0),rho1*np.exp(1j*phi1))
        k2 = np.zeros(self.u0.shape)
        k2[rho0>0] = np.imag( np.exp(-1j*phi0[rho0>0]) * ( self.NL_1(U0,self.A0)[rho0>0] + self.NL_1(U1,self.B0)[rho0>0])  / rho0[rho0>0]) #+ self.f0
        p2 = np.zeros(self.u1.shape)
        p2[rho1>0] = np.imag( np.exp(-1j*phi1[rho1>0]) * ( self.NL_2(U0,U1,self.A1)[rho1>0] + self.NL_2(U1,U0,self.B1)[rho1>0])  / rho1[rho1>0]) #+ self.f1

        ##### k3: f(t+h/2, w + h*k2/2)
        phi0 = (np.angle(self.u0) + k2*self.dt/2)
        phi1 = (np.angle(self.u1) + p2*self.dt/2)
        [U0,U1] = self.roll_fields(rho0*np.exp(1j*phi0),rho1*np.exp(1j*phi1))
        k3 = np.zeros(self.u0.shape)
        k3[rho0>0] = np.imag( np.exp(-1j*phi0[rho0>0]) * ( self.NL_1(U0,self.A0)[rho0>0] + self.NL_1(U1,self.B0)[rho0>0])  / rho0[rho0>0]) #+ self.f0
        p3 = np.zeros(self.u1.shape)
        p3[rho1>0] = np.imag( np.exp(-1j*phi1[rho1>0]) * ( self.NL_2(U0,U1,self.A1)[rho1>0] + self.NL_2(U1,U0,self.B1)[rho1>0])  / rho1[rho1>0]) #+ self.f1

        ##### k4: f(t+h, w + h * k3)
        phi0 = (np.angle(self.u0) + k3*self.dt) 
        phi1 = (np.angle(self.u1) + p3*self.dt) 
        [U0,U1] = self.roll_fields(rho0*np.exp(1j*phi0),rho1*np.exp(1j*phi1))
        k4 = np.zeros(self.u0.shape)
        k4[rho0>0] = np.imag( np.exp(-1j*phi0[rho0>0]) * ( self.NL_1(U0,self.A0)[rho0>0] + self.NL_1(U1,self.B0)[rho0>0])  / rho0[rho0>0]) #+ self.f0
        p4 = np.zeros(self.u1.shape)
        p4[rho1>0] = np.imag( np.exp(-1j*phi1[rho1>0]) * ( self.NL_2(U0,U1,self.A1)[rho1>0] + self.NL_2(U1,U0,self.B1)[rho1>0])  / rho1[rho1>0]) #+ self.f1

        # Update final step
        phi0 = np.angle(self.u0) + (self.dt/6) * ( k1 + 2*k2 + 2*k3 + k4)
        phi1 = np.angle(self.u1) + (self.dt/6) * ( p1 + 2*p2 + 2*p3 + p4)
        self.u0 = rho0*np.exp(1j*phi0)
        self.u1 = rho1*np.exp(1j*phi1)
        
        # Update time
        self.t += self.dt
        self.tstep += 1
        
        return

    def step_phase_only_HD(self):     
        """
        Take a time-step, but only advance the phases and keep the magnitudes the same. Because viscosity doesn't play a role, we remove the linear operator. 
        Dynamical inputs needed: u0, u1. Returns nothing, just updates [u0,u1].
        """
        ## The magnitudes are fixed, so we will define rho0 and rho1 and keep them fixed.
        rho0 = np.abs(self.u0)
        
        ########
        #### RK4 
        # Intermediate steps    
        ##### k1: f(t,w)
        # Only updating the phase, so we need to apply Imag[exp(-1j*phi)*f / rho].
        phi0 = np.angle(self.u0)
        U0 = self.roll_fields_HD(rho0*np.exp(1j*phi0))
        k1 = np.zeros(self.u0.shape)
        k1[rho0>0] = np.imag( np.exp(-1j*phi0[rho0>0]) * ( self.NL_1(U0,self.A0)[rho0>0] )  / rho0[rho0>0]) #+ self.f0
        
        ##### k2: f(t+h/2, w + h*k1/2)
        phi0 = (np.angle(self.u0) + k1*self.dt/2)
        U0 = self.roll_fields_HD(rho0*np.exp(1j*phi0))
        k2 = np.zeros(self.u0.shape)
        k2[rho0>0] = np.imag( np.exp(-1j*phi0[rho0>0]) * ( self.NL_1(U0,self.A0)[rho0>0] )  / rho0[rho0>0]) #+ self.f0

        ##### k3: f(t+h/2, w + h*k2/2)
        phi0 = (np.angle(self.u0) + k2*self.dt/2)
        U0 = self.roll_fields_HD(rho0*np.exp(1j*phi0))
        k3 = np.zeros(self.u0.shape)
        k3[rho0>0] = np.imag( np.exp(-1j*phi0[rho0>0]) * ( self.NL_1(U0,self.A0)[rho0>0] )  / rho0[rho0>0]) #+ self.f0

        ##### k4: f(t+h, w + h * k3)
        phi0 = (np.angle(self.u0) + k3*self.dt) 
        U0 = self.roll_fields_HD(rho0*np.exp(1j*phi0))
        k4 = np.zeros(self.u0.shape)
        k4[rho0>0] = np.imag( np.exp(-1j*phi0[rho0>0]) * ( self.NL_1(U0,self.A0)[rho0>0] )  / rho0[rho0>0]) #+ self.f0

        # Update final step
        phi0 = np.angle(self.u0) + (self.dt/6) * ( k1 + 2*k2 + 2*k3 + k4)
        self.u0 = rho0*np.exp(1j*phi0)
        
        # Update time
        self.t += self.dt
        self.tstep += 1
        
        return
    
    def step_old(self):     
        """
        Take a time-step. Dynamical inputs needed: u0. Returns nothing, just updates u0.

        """
        
        #### RK4 
        # Intermediate steps    
        # k1: f(t,w)
        self.f0[int(self.nf):int(self.nf+2)] = self.eps0/np.conj(2*self.u0[int(self.nf):int(self.nf+2)])
        k1 = ( self.f0 + self.NL_1(self.u0,self.A0)) 

        # k2: f(t+h/2, w + h*k1/2)
        wtemp0 = np.exp(self.L0*(self.dt/2)) * (self.u0 + k1*self.dt/2)
        self.f0[int(self.nf):int(self.nf+2)] = self.eps0/np.conj(2*wtemp0[int(self.nf):int(self.nf+2)])     
        k2 = np.exp(-self.L0*(self.dt/2)) * ( self.f0 + self.NL_1(wtemp0,self.A0)) 

        # k3: f(t+h/2, w + h*k2/2)
        wtemp0 = np.exp(self.L0*(self.dt/2)) * (self.u0 + k2*self.dt/2)
        self.f0[int(self.nf):int(self.nf+2)] = self.eps0/np.conj(2*wtemp0[int(self.nf):int(self.nf+2)])
        k3 = np.exp(-self.L0*(self.dt/2)) * ( self.f0 + self.NL_1(wtemp0, self.A0)) 

        # k4: f(t+h, w + h * k3)
        wtemp0 = np.exp(self.L0*(self.dt)) * (self.u0 + k3*self.dt) 
        self.f0[int(self.nf):int(self.nf+2)] = self.eps0/np.conj(2*wtemp0[int(self.nf):int(self.nf+2)]) 
        k4 = np.exp(-self.L0*(self.dt)) * ( self.f0 + self.NL_1( wtemp0, self.A0))

        # Update final step
        self.u0 = np.exp(self.L0*self.dt)*self.u0 + np.exp(self.L0*self.dt) * (self.dt/6) * ( k1 + 2*k2 + 2*k3 + k4)

        # Update time
        self.t += self.dt
        self.tstep += 1
        
        return
        
    #########################################
    ####       Import/Export      ###########
    #########################################
 
    def get_state(self):
        """
        Get current state of model: returns [tstep,t,u0,u1]
        """
        return [self.tstep,self.t,self.u0,self.u1]
    
    def set_state(self,data):
        """
        Set current state of model: input must be in format [tstep,t,u0,u1]. To be used with 'load_data'. 
        No need to use set_state unless you want to manually create a dataset.
        """
        [self.tstep,self.t,self.u0,self.u1] = data
        return
    
    def load_data(self,name,num = -1):
        """
        Imports .h5 file with given name and sets the state of the model. Note that you can also manually set the state by calling the 
        'set_state' fuction.
        """
        # Searches for last non-nan save. (In case of blow-up.)
        if num == -1:
            is_nan = True
            num_tmp = 0
            while is_nan:
                num_tmp -= 1
                with h5py.File(name,'r') as f:
                    self.tstep = f['state']['tstep'][num_tmp]
                    self.t = f['state']['t'][num_tmp]
                    self.u0 = f['state']['u0'][num_tmp]
                    self.u1 = f['state']['u1'][num_tmp]
                is_nan = np.any(np.isnan(self.u0))
        else:
            with h5py.File(name,'r') as f:
                self.tstep = f['state']['tstep'][num]
                self.t = f['state']['t'][num]
                self.u0 = f['state']['u0'][num]
                self.u1 = f['state']['u1'][num]

        return 

    def export_name(self,today):
        nfstr = str(self.nf).replace(".", "d")
#         b0str = str(self.A0[1]).replace(".", "d")
#         b1str = str(self.A1[1]).replace(".", "d")
        eps1str = str(self.eps1).replace(".", "d")
        return 'shell_data_N_'+str(self.N)+'_nf_'+nfstr+'_eps1_'+eps1str+'_'+str(today)

    def get_params(self):
        """
        Get parameters of model: returns [nf,N,lam,dt,A0,B0,A1,B1,eps0,eps1]
        """
        return {'nf':self.nf, 'N':self.N,'lam':self.lam, 'dt':self.dt,'A0':self.A0,'B0':self.B0,'A1': self.A1,'B1':self.B1, 'eps0':self.eps0,'eps1':self.eps1}
    
    def get_scalars(self):
        """
        Get scalar outputs of model: returns ['tstep','t','en_u0','en_u1','diss_u0','diss_u1','hdiss_u0','hdiss_u1','h_u0','h_u1','diss_A','hdiss_A','en_u0_nf','en_u1_nf']
        """
        # Calculate exponent for H_2D and H_3D conservation
        r2D = -np.log(np.abs(-self.A0[1]-1))/np.log(self.lam)
        r3D = np.log(-self.A1[0]/self.B1[2])/np.log(self.lam)/2
        
        return [
            self.tstep,
            self.t,
            self.energy(self.u0,0),
            self.energy(self.u1,0),
            self.nu*self.energy(self.u0,2),
            self.nu*self.energy(self.u1,2) + self.nu*self.Q**2*self.energy(self.u1,0),
            self.nu_l*np.sum(np.real(self.u0[:2]*np.conj(self.u0)[:2])),
            self.nu_l*np.sum(np.real(self.u1[:2]*np.conj(self.u1)[:2])),
            self.energy(self.u0,r2D),
            self.energy(self.u1,r3D), # For sign-definite A^2 = k^{-2} u1^2
            self.nu*self.energy(self.u1,2+r3D),
            self.nu_l*np.sum(np.real(self.k[:2]**(r3D)*self.u1[:2]*np.conj(self.u1)[:2])),
            np.real(self.u0[self.nf]*np.conj(self.u0[self.nf])),
            np.real(self.u1[self.nf]*np.conj(self.u1[self.nf])),
        ]

    def get_scalars_2(self):
        """
        Get scalar outputs of model: returns ['tstep','t','flux0','flux1','fluxA','thetauuu_IC_m2','thetauuu_IC_m1','thetauuu_IC','thetauuu_IC_p1','thetauuu_IC_p2','thetabbu_IC_m2','thetabbu_IC_m1','thetabbu_IC','thetabbu_IC_p1','thetabbu_IC_p2','thetabub_IC_m2','thetabub_IC_m1','thetabub_IC','thetabub_IC_p1','thetabub_IC_p2','thetaubb_IC_m2','thetaubb_IC_m1','thetaubb_IC','thetaubb_IC_p1','thetaubb_IC_p2','thetauuu_FC_m2','thetauuu_FC_m1','thetauuu_FC','thetauuu_FC_p1','thetauuu_FC_p2','thetabbu_FC_m2','thetabbu_FC_m1','thetabbu_FC','thetabbu_FC_p1','thetabbu_FC_p2','thetabub_FC_m2','thetabub_FC_m1','thetabub_FC','thetabub_FC_p1','thetabub_FC_p2','thetaubb_FC_m2','thetaubb_FC_m1','thetaubb_FC','thetaubb_FC_p1','thetaubb_FC_p2','delta_IC_m2','delta_IC_m1','delta_IC','delta_IC_p1','delta_IC_p2','delta_FC_m2','delta_FC_m1','delta_FC','delta_FC_p1','delta_FC_p2']
        """
        # Calculate exponent for H_2D and H_3D conservation
        r3D = np.log(-self.A1[0]/self.B1[2])/np.log(self.lam)/2

        # Calculate the triad phases
        [U0,U1] = self.roll_fields(self.u0,self.u1)
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
        delta = phi1k - phi0k
        
        thetauuu = thetauuu -2*np.pi*np.round(thetauuu/np.pi/2) # From [-pi,pi]
        thetabbu = thetabbu -2*np.pi*np.round(thetabbu/np.pi/2) # From [-pi,pi]
        thetabub = thetabub -2*np.pi*np.round(thetabub/np.pi/2) # From [-pi,pi]
        thetaubb = thetaubb -2*np.pi*np.round(thetaubb/np.pi/2) # From [-pi,pi]
        delta = delta -2*np.pi*np.round(delta/np.pi/2) # From [-pi,pi]
        
        return [
            self.tstep,
            self.t,
            self.Pi_0(U0,self.A0)[int(self.nf/2)],
            self.Pi_1(U0,U1,self.B0,self.A1,self.B1)[int(self.nf/2)],
            self.Pi_A(U0,U1,self.B0,self.A1,self.B1,r3D)[int(self.nf/2)],
            thetauuu[int(self.nf/2-2)],
            thetauuu[int(self.nf/2-1)],
            thetauuu[int(self.nf/2)],
            thetauuu[int(self.nf/2+1)],
            thetauuu[int(self.nf/2+2)],
            thetabbu[int(self.nf/2-2)],
            thetabbu[int(self.nf/2-1)],
            thetabbu[int(self.nf/2)],
            thetabbu[int(self.nf/2+1)],
            thetabbu[int(self.nf/2+2)],
            thetabub[int(self.nf/2-2)],
            thetabub[int(self.nf/2-1)],
            thetabub[int(self.nf/2)],
            thetabub[int(self.nf/2+1)],
            thetabub[int(self.nf/2+2)],
            thetaubb[int(self.nf/2-2)],
            thetaubb[int(self.nf/2-1)],
            thetaubb[int(self.nf/2)],
            thetaubb[int(self.nf/2+1)],
            thetaubb[int(self.nf/2+2)],
            thetauuu[int(self.nf+(self.N-self.nf)/4-2)],
            thetauuu[int(self.nf+(self.N-self.nf)/4-1)],
            thetauuu[int(self.nf+(self.N-self.nf)/4)],
            thetauuu[int(self.nf+(self.N-self.nf)/4+1)],
            thetauuu[int(self.nf+(self.N-self.nf)/4+2)],
            thetabbu[int(self.nf+(self.N-self.nf)/4-2)],
            thetabbu[int(self.nf+(self.N-self.nf)/4-1)],
            thetabbu[int(self.nf+(self.N-self.nf)/4)],
            thetabbu[int(self.nf+(self.N-self.nf)/4+1)],
            thetabbu[int(self.nf+(self.N-self.nf)/4+2)],
            thetabub[int(self.nf+(self.N-self.nf)/4-2)],
            thetabub[int(self.nf+(self.N-self.nf)/4-1)],
            thetabub[int(self.nf+(self.N-self.nf)/4)],
            thetabub[int(self.nf+(self.N-self.nf)/4+1)],
            thetabub[int(self.nf+(self.N-self.nf)/4+2)],
            thetaubb[int(self.nf+(self.N-self.nf)/4-2)],
            thetaubb[int(self.nf+(self.N-self.nf)/4-1)],
            thetaubb[int(self.nf+(self.N-self.nf)/4)],
            thetaubb[int(self.nf+(self.N-self.nf)/4+1)],
            thetaubb[int(self.nf+(self.N-self.nf)/4+2)],
            delta[int(self.nf/2-2)],
            delta[int(self.nf/2-1)],
            delta[int(self.nf/2)],
            delta[int(self.nf/2+1)],
            delta[int(self.nf/2+2)],
            delta[int(self.nf+(self.N-self.nf)/4-2)],
            delta[int(self.nf+(self.N-self.nf)/4-1)],
            delta[int(self.nf+(self.N-self.nf)/4)],
            delta[int(self.nf+(self.N-self.nf)/4+1)],
            delta[int(self.nf+(self.N-self.nf)/4+2)]
        ]

    def get_scalars_3(self):
        """
        Get scalar outputs of model: returns ['tstep','t','theta_out[N/4]','coeff_self[N/4]','theta_rest[N/4]','theta_out[N/3]','coeff_self[N/3]','theta_rest[N/3]','theta_out[N/2]','coeff_self[N/2]','theta_rest[N/2]','theta_out[2N/3]','coeff_self[2N/3]','theta_rest[2N/3]','theta_out[3N/4]','coeff_self[3N/4]','theta_rest[3N/4]']
        """
        # Roll fields and ks
        U0 = self.roll_fields_HD(self.u0)
        u0p3 = np.roll(self.u0,-3)
        u0p3[-3:] = 0.0
        u0p4 = np.roll(self.u0,-4)
        u0p4[-4:] = 0.0
        kp2 = np.roll(self.k,-2)
        kp3 = np.roll(self.k,-3)

        # Calculate the triad phases
        phi0k = np.angle(U0[2])
        phi0kp1 = np.angle(U0[3])
        phi0kp2 = np.angle(U0[4])
        thetauuu = (phi0kp2-phi0kp1-phi0k)
        thetauuu = thetauuu -2*np.pi*np.round(thetauuu/np.pi/2) # From [-pi,pi]

        # Roll phases
        thetauuu_p2 = np.roll(thetauuu,-2)
        thetauuu_p2[-2:] = 0.0
        thetauuu_p1 = np.roll(thetauuu,-1)
        thetauuu_p1[-1:] = 0.0
        thetauuu_m1 = np.roll(thetauuu,1)
        # Theta_-1 = phi_1 - phi_0
        # Theta_-2 = phi_0
        thetauuu_m1[0] = phi0k[1]-phi0k[0]
        thetauuu_m2 = np.roll(thetauuu,2)
        thetauuu_m2[0] = phi0k[0]
        thetauuu_m2[1] = phi0k[1]-phi0k[0]
     
        # Full term
        coeff_self = np.zeros(self.N)
        theta_out = np.zeros(self.N)
        theta_out[:] = np.copy(thetauuu[:])
        coeff_self[:-2] = -self.kp1[:-2]*(self.A0[2]*(np.abs(U0[2][:-2])*np.abs(U0[3][:-2]))/(np.abs(U0[4][:-2])) + self.A0[1]*(np.abs(U0[2][:-2])*np.abs(U0[4][:-2]))/(np.abs(U0[3][:-2])) + self.A0[0]*(np.abs(U0[4][:-2])*np.abs(U0[3][:-2]))/(np.abs(U0[2][:-2])))

        # Theta n+2
        term_rest_p2 = np.zeros(self.N)
        term_rest_p2[:-4] = kp3[:-4]*self.A0[0]*((np.abs(u0p3[:-4])*np.abs(u0p4[:-4]))/(np.abs(U0[4][:-4])))*np.cos(thetauuu_p2[:-4])

        # Theta n+1
        term_rest_p1 = np.zeros(self.N)
        term_rest_p1[:-3] = kp2[:-3]*np.abs(u0p3[:-3])*(self.A0[1]*(np.abs(U0[3][:-3]))/(np.abs(U0[4][:-3])) - self.A0[0]*(np.abs(U0[4][:-3]))/(np.abs(U0[3][:-3])))*np.cos(thetauuu_p1[:-3])

        # Theta n-1
        term_rest_m1 = np.zeros(self.N)
        term_rest_m1[1:-1] = self.k[1:-1]*np.abs(U0[1][1:-1])*(self.A0[2]*(np.abs(U0[2][1:-1]))/(np.abs(U0[3][1:-1])) - self.A0[1]*(np.abs(U0[3][1:-1]))/(np.abs(U0[2][1:-1])))*np.cos(thetauuu_m1[1:-1])
        
        # Theta n-2
        term_rest_m2 = np.zeros(self.N)
        term_rest_m2[2:] = self.km1[2:]*self.A0[2]*((np.abs(U0[0][2:])*np.abs(U0[1][2:]))/(np.abs(U0[2][2:])))*np.cos(thetauuu_m2[2:])
        
        term_rest = term_rest_p2 + term_rest_p1 + term_rest_m1 + term_rest_m2
        
        return [
            self.tstep,
            self.t,
            theta_out[self.N//4],
            coeff_self[self.N//4],
            term_rest[self.N//4],
            theta_out[self.N//3],
            coeff_self[self.N//3],
            term_rest[self.N//3],
            theta_out[self.N//2],
            coeff_self[self.N//2],
            term_rest[self.N//2],
            theta_out[2*self.N//3],
            coeff_self[2*self.N//3],
            term_rest[2*self.N//3],
            theta_out[3*self.N//4],
            coeff_self[3*self.N//4],
            term_rest[3*self.N//4],
        ]
                       
    def export_state(self,odir,today=date.today(),overwrite=True):
        """
        Inputs: name (name of file), odir (output directory), overwrite (=True by default), if True, then regardless 
        of if there is already a file there or not, it'll overwrite that file. Otherwise, it'll append to the 
        currently existing file. If there is no file, then it will create one.

        Exports odir+ self.export_name() +'_state.h5' file, 
        which contains two groups: 
            1) 'parameters' (depends on the mode)
            2) 'state' [tstep,t,u0,u1]
        into directory 'odir'.
        """
        fname = odir+ self.export_name(today) +'_state.h5'
        if not path.exists(fname):
            with h5py.File(fname,'w') as f:
                # Parameters
                params = f.create_group('parameters')
                paramdict = self.get_params()
                for k, v in paramdict.items():
                    params.create_dataset(k, data=np.array(v,dtype=np.float64))

                # State of simulation
                state = f.create_group('state')
                state.create_dataset('tstep', data = [self.tstep],maxshape=(None,),chunks=True)
                state.create_dataset('t', data = [self.t],maxshape=(None,),chunks=True)
                state.create_dataset('u0', data = [self.u0],maxshape=(None,np.shape(self.u0)[0]),chunks=True)
                state.create_dataset('u1', data = [self.u1],maxshape=(None,np.shape(self.u1)[0]),chunks=True)
        else:
            if overwrite:
                with h5py.File(fname,'w') as f:
                    # Parameters
                    params = f.create_group('parameters')
                    paramdict = self.get_params()
                    for k, v in paramdict.items():
                        params.create_dataset(k, data=np.array(v,dtype=np.float64))

                    # State of simulation
                    state = f.create_group('state')
                    state.create_dataset('tstep', data = [self.tstep],maxshape=(None,),chunks=True)
                    state.create_dataset('t', data = [self.t],maxshape=(None,),chunks=True)
                    state.create_dataset('u0', data = [self.u0],maxshape=(None,np.shape(self.u0)[0]),chunks=True)
                    state.create_dataset('u1', data = [self.u1],maxshape=(None,np.shape(self.u1)[0]),chunks=True)
            else:
                with h5py.File(fname,'a') as f:
                    state = f['state']
                    state['tstep'].resize((state['tstep'].shape[0] + 1), axis = 0)
                    state['tstep'][-1:] = [self.tstep]
                    state['t'].resize((state['t'].shape[0] + 1), axis = 0)
                    state['t'][-1:] = [self.t]
                    state['u0'].resize((state['u0'].shape[0] + 1), axis = 0)
                    state['u0'][-1:] = [self.u0]
                    state['u1'].resize((state['u1'].shape[0] + 1), axis = 0)
                    state['u1'][-1:] = [self.u1]

        return

    def export_scalars(self,odir,data,today=date.today(),overwrite=True,name=None):
        """
        Inputs: 
         - odir (output directory)
         - data (data, based on appending self.get_scalars() in loop)
         - overwrite (=True by default), if True, then regardless 
        of if there is already a file there or not, it'll overwrite that file. 
        Otherwise, it'll append to the currently existing file

        Exports self.export_name()+'.h5' file, which contains two groups: 
            1) 'parameters' (depends on the mode)
            2) 'scalars' (depends on the mode) 
        into directory 'odir'.
        """
        if name==None:
            fname = odir+ self.export_name(today) +'_scalars.h5'
        else:
            fname = odir+ self.export_name(today) +'_scalars_'+name+'.h5'
        with h5py.File(fname,'w') as f:
            # Parameters
            params = f.create_group('parameters')
            for k, v in self.get_params().items():
                params.create_dataset(k, data=np.array(v,dtype=np.float64))

            scalars = f.create_group('scalars')
            for ii,d in enumerate(np.array(data).T):
                scalars.create_dataset(self.okeys[ii],data=np.array(d))
                
        # if not path.exists(fname):
        #     with h5py.File(fname,'w') as f:
        #         # Parameters
        #         params = f.create_group('parameters')
        #         for k, v in self.get_params().items():
        #             params.create_dataset(k, data=np.array(v,dtype=np.float64))

        #         scalars = f.create_group('scalars')
        #         for ii,d in enumerate(np.array(data).T):
        #             scalars.create_dataset(self.okeys[ii],data=np.array(d))
        
        # else:
        #     if overwrite:
        #         with h5py.File(fname,'w') as f:
        #             params = f.create_group('parameters')
        #             for k, v in self.get_params().items():
        #                 params.create_dataset(k, data=np.array(v,dtype=np.float64))

        #             scalars = f.create_group('scalars')
        #             for ii,d in enumerate(np.array(data).T):
        #                 scalars.create_dataset(self.okeys[ii],data=np.array(d))
        #     else:
        #         with h5py.File(fname,'a') as f:
        #             for ii,d in enumerate(np.array(data).T):
        #                 del f['scalars'][self.okeys[ii]]
        #                 f['scalars'][self.okeys[ii]] = np.array(d)

        return

    def export_scalars_2(self,odir,data,today=date.today(),overwrite=True,name=None):
        """
        Inputs: 
         - odir (output directory)
         - data (data, based on appending self.get_scalars() in loop)
         - overwrite (=True by default), if True, then regardless 
        of if there is already a file there or not, it'll overwrite that file. 
        Otherwise, it'll append to the currently existing file

        Exports self.export_name()+'.h5' file, which contains two groups: 
            1) 'parameters' (depends on the mode)
            2) 'scalars' (depends on the mode) 
        into directory 'odir'.
        """
        if name==None:
            fname = odir+ self.export_name(today) +'_scalars_2.h5'
        else:
            fname = odir+ self.export_name(today) +'_scalars_2_'+name+'.h5'
        with h5py.File(fname,'w') as f:
            # Parameters
            params = f.create_group('parameters')
            for k, v in self.get_params().items():
                params.create_dataset(k, data=np.array(v,dtype=np.float64))

            scalars = f.create_group('scalars')
            for ii,d in enumerate(np.array(data).T):
                scalars.create_dataset(self.okeys_2[ii],data=np.array(d))

        return

    def export_scalars_3(self,odir,data,today=date.today(),overwrite=True,name=None):
        """
        Inputs: 
         - odir (output directory)
         - data (data, based on appending self.get_scalars() in loop)
         - overwrite (=True by default), if True, then regardless 
        of if there is already a file there or not, it'll overwrite that file. 
        Otherwise, it'll append to the currently existing file

        Exports self.export_name()+'.h5' file, which contains two groups: 
            1) 'parameters' (depends on the mode)
            2) 'scalars' (depends on the mode) 
        into directory 'odir'.
        """
        if name==None:
            fname = odir+ self.export_name(today) +'_scalars_3.h5'
        else:
            fname = odir+ self.export_name(today) +'_scalars_3_'+name+'.h5'
        with h5py.File(fname,'w') as f:
            # Parameters
            params = f.create_group('parameters')
            for k, v in self.get_params().items():
                params.create_dataset(k, data=np.array(v,dtype=np.float64))

            scalars = f.create_group('scalars')
            for ii,d in enumerate(np.array(data).T):
                scalars.create_dataset(self.okeys_3[ii],data=np.array(d))

        return