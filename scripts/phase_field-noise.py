#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main function that is called to initialize and run phase-field dynamics
"""

from __future__ import print_function
import fipy as fp
import os
#from fipy.solvers.pysparse import LinearLUSolver as Solver
import numpy as np
np.set_printoptions(threshold=np.inf)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import argparse
from utils.input_parse import *
from utils.graphics import *
import utils.free_energy as f_en
import utils.sim_tools as sim_tools
import timeit
import subprocess
import cv2
import h5py
from scipy.interpolate import griddata


def return_noise(mesh,time_step,noise_strength):
    """
        return_noise returns Gaussian noise of a given variance = noise_strength
        for a particular mesh & time_step
    """
    p = fp.GaussianNoiseVariable(mesh=mesh,variance=noise_strength/(time_step*mesh.cellVolumes))    
    p = p - np.mean(p.value);
    return(p);
    
def run_CH(args):
    """
    Function takes in path to input params, output_folder, and optionally params file that
    are passed while running code. With these parameters, this function initializes and runs
    phase-field simulations while writing output to files.

    **Input variables**

    -   args.i = Path to input params files (required)
    -   args.o = path to output folder (required)
    -   args.p = path to parameter file (optional)
    -   args.pN     =   Nth parameter to use from input (optional)

    """

    # In[4]:
    seed = np.random.randint(1e8);
    input_parameters = input_parse(args.i);

    if args.p is not None:
        params = input_parse(args.p,params_flag=True)
        print(params)
        par_name = str(list(params.keys())[0])
        par_values = params[par_name];
        if args.pN is not None:
            par_values = [par_values[int(args.pN)-1]];
    else:
        par_name = 'nx';
        par_values = [(int(input_parameters['nx']))];
    for par in par_values:
        start = timeit.default_timer()

        input_parameters[par_name] = par;
        nx = int(input_parameters['nx'])
        dx = input_parameters['dx']
        c_alpha = input_parameters['c_alpha'];
        c_beta = input_parameters['c_beta'];
        kappa = input_parameters['kappa']
        M_protein = input_parameters['M_protein']
        M_rna = input_parameters['M_rna']
        dimension = int(input_parameters['dimension'])
        plot_flag = bool(input_parameters['plot_flag'])
        rho_s = input_parameters['rho_s']
        rho_r = input_parameters['rho_r']
        chi = input_parameters['chi']
        changing_chi = int(input_parameters['changing_chi']);
        fh = int(input_parameters['fh']);

        """ Unpack rates_1 from the parameters """
        k_production = input_parameters['k_production']
        k_degradation = input_parameters['k_degradation'];

        """ Define size of initial nucleus """
        nucleus_size = int(input_parameters['nucleus_size']);
        if 'protein_nucleus_location' in input_parameters.keys():
            protein_nucleus_location = input_parameters['protein_nucleus_location']
        else:
            protein_nucleus_location = (0,0)
        
        """ Initial RNA & protein concentrations """
        phi_p_0 = input_parameters['phi_p_0'];
        phi_r_0 = input_parameters['phi_r_0'];
        
        """Load coordinates and stds of RNA production locations"""
        coordinates = input_parameters['coordinates']
        std = input_parameters['std']
        # rescale coordinates and std to match over different dx * nx ratios
        #coordinates = [(int(x[0] / dx), int(x[1] / dx)) for x in coordinates]
        #std = [x / dx for x in std]
        
        localization_scale = input_parameters['localization_scale']
        
        
        if 'a' in input_parameters.keys():
            a = float(input_parameters['a']);
        else:
            a=0.0;
        if 'b' in input_parameters.keys():
            b = float(input_parameters['b']);
        else:
            b=0.0;
        if 'c' in input_parameters.keys():
            c = float(input_parameters['c']);
        else:
            c=0.0;
            
        if 'noise_strength' in input_parameters.keys():
            noise_strength = float(input_parameters['noise_strength']);
        else:
            noise_strength=0.0;

        if 'm1' in input_parameters.keys():
            m1 = float(input_parameters['m1']);
        else:
            m1=1.0;
 
        if 'kp_noise' in input_parameters.keys():
            kp_noise = float(input_parameters['kp_noise']);
        else:
            kp_noise=0.0

        if 'chi_ps' in input_parameters.keys():
            chi_ps = float(input_parameters['chi_ps'])
        else:
            chi_ps=None;
            
        if 'chi_rs' in input_parameters.keys():
            chi_rs = float(input_parameters['chi_rs'])
        else:
            chi_rs=0.0;
            
        if 'distribution' in input_parameters.keys():
            distribution = input_parameters['distribution']
        else:
            distribution = 0 # this is'gaussian'
            
        if 'r' in input_parameters.keys():
            r = float(input_parameters['r']);
        else:
            r=1.0;
        if isinstance(coordinates, float):
            type_ = 'gradient'
        elif isinstance(coordinates, list):
            if len(coordinates) != 0:
                type_ = 'gaussian'
            else:
                type_ = 'unlocalized'


            
        if 'seed' in input_parameters.keys():
            seed=int(input_parameters['seed'])

        fp.numerix.random.seed(seed);

        """Define the mesh"""
        
        if dimension==2:
            if int(input_parameters['circ_flag']):
                mesh = sim_tools.create_circular_mesh(radius=float(nx)*dx/2,cellSize=dx*1.5)
            else:
                mesh = fp.Grid2D(nx=nx, ny=nx, dx=dx, dy=dx)
                mesh = mesh-float(nx)*dx*0.5
        elif dimension==3:
            mesh = fp.Grid3D(nx=nx, ny=nx,nz=nx, dx=dx, dy=dx,dz=dx)
            mesh = mesh-float(nx)*dx*0.5
        
        
        """
        Set-up the appropriate choice of free-energy
            fh is a flag for employing Flory-Huggins instead of double-well
            changing_chi ==2 uses the gaussian form & 1 == uses double-well LG expression
            changing_chi ==0 is not changing_chi and there for backwards compatability
            rho_s/rho_r is height of double-well potential for protein/RNA respectively
            kappa is surface tension parameter for protein
            chi is value of pairwise interaction
            Y is value of landau-ginzburg like three-way interaction
            mu_r chooses whether you use D-R (mu_r=0) or chemical potential fo RNA (mu_r=1)
            a,ratio, and p characterize the gaussian form of chi
        """

        if not fh:

            if changing_chi==2:
                FE = f_en.free_energy_changing_chi(c_alpha=c_alpha,c_beta=c_beta,
                                                   rho_s=rho_s,rho_r=rho_r,chi=chi,kappa=kappa,a=input_parameters['a'],
                                                   ratio=input_parameters['ratio'],p=input_parameters['p'])
            elif changing_chi==1:
                FE = f_en.free_energy_changing_chi_LG(c_alpha=c_alpha,c_beta=c_beta,rho_s=rho_s,rho_r=rho_r ,chi=chi,kappa=kappa,a=a,b=b,c=c)
            else:
                FE = f_en.free_energy(c_alpha=c_alpha,c_beta=c_beta,rho_s=rho_s,chi=chi,kappa=kappa)

        else:

            if changing_chi==2:
                FE = f_en.free_energy_FH_changing_chi(c_alpha=c_alpha,c_beta=c_beta,
                                                      rho_s=rho_s,chi=chi,kappa=kappa,a=input_parameters['a'],
                                                      ratio=input_parameters['ratio'],p=input_parameters['p']);
            elif changing_chi==1:
                FE = f_en.free_energy_FH_changing_chi_LG(c_alpha=c_alpha,c_beta=c_beta,
                                                         rho_s=rho_s,rho_r=rho_r,chi=chi,
                                                         kappa=kappa,a=a,b=b,c=c,chi_ps=chi_ps,chi_rs=chi_rs,r=r)
            else:
                FE = f_en.free_energy_FH(c_alpha=c_alpha,c_beta=c_beta,rho_s=rho_s,chi=chi,kappa=kappa);

        """
        Define the parameters that dictate reaction kinetics
            if multiplier is specified, so must  t_change.
            Then after t_change has passed, the simulation will multiply k_production by multiplier
            threshold will ensure production only at phi_p>=threshold values
        """


        if 'multiplier' in input_parameters.keys():
            rates_1 = f_en.RNA_reactions(k_production=k_production,
                                       k_degradation=k_degradation,
                                       mesh=mesh,
                                       coordinates=coordinates,
                                       std=std,
                                       distribution=distribution,
                                       threshold=input_parameters['threshold'],
                                       t_change=input_parameters['t_change'],
                                       multiplier=input_parameters['multiplier'],m1=m1,kp_noise=kp_noise,
                                       scale=localization_scale);
            
            #rates_2 = f_en.RNA_reactions(k_production=10.0,
#                                        k_degradation=0.5,
#                                        mesh=mesh,
#                                        coordinates=[(-8,0)],
#                                        std=[4],
#                                        threshold=input_parameters['threshold'],
#                                        t_change=input_parameters['t_change'],
#                                        multiplier=input_parameters['multiplier'],m1=m1,kp_noise=kp_noise,
#                                        scale=localization_scale);
        else:
            rates_1 = f_en.RNA_reactions(k_production=k_production,k_degradation=k_degradation,
                                       mesh=mesh,
                                       coordinates=coordinates,
                                       std=std,
                                       threshold=input_parameters['threshold'],m1=m1,kp_noise=kp_noise,
                                       scale=localization_scale);




        phi_p = fp.CellVariable(mesh=mesh, name=r'$\phi_{prot}$', hasOld=True,value = phi_p_0)
        phi_r = fp.CellVariable(mesh=mesh, name=r'$\phi_{RNA}$', hasOld=True,value = phi_r_0)
        
        # Two new classes of RNA molecules
        #phi_r_1 = fp.CellVariable(mesh=mesh, name=r'$\phi_{RNA_1}$', hasOld=True,value = phi_r_0)
        #phi_r_2 = fp.CellVariable(mesh=mesh, name=r'$\phi_{RNA_2}$', hasOld=True,value = phi_r_0)
        
        
        phi_p[:] =fp.GaussianNoiseVariable(mesh=mesh,mean=phi_p_0,variance=0.5*phi_p_0).value
        phi_p[phi_p<phi_p_0*0.5] = phi_p_0*0.5;
        phi_p[phi_p>phi_p_0*1.5] = phi_p_0*1.5;
        print(min(phi_p),max(phi_p),np.mean(phi_p))


        phi_r[:] =fp.GaussianNoiseVariable(mesh=mesh,mean=phi_r_0,variance=0.0*phi_r_0).value
        phi_r[phi_r<phi_r_0*0.5] = phi_r_0*0.5;
        phi_r[phi_r>phi_r_0*1.5] = phi_r_0*1.5;

        print(min(phi_r),max(phi_r),np.mean(phi_r))

        # # We nucleate a high dense region at the center of the grid
        # # array of sample $\phi_{a}$-values:

        # In[5]:


        sim_tools.nucleate_seed(mesh,phi_p,
                                phia_value=0.9*(c_beta),
                                nucleus_size=nucleus_size,
                                dimension=dimension, 
                                location=protein_nucleus_location)



        # ## Define relevant equations for this system

        # In[6]:
        t = fp.Variable(0.0)
        dt = input_parameters['dt'];
        dt_max = input_parameters['dt_max'];
        dt_min = input_parameters['dt_min'];
        tolerance = input_parameters['tolerance'];
        total_steps = int(input_parameters['total_steps']);
        checkpoint = int(input_parameters['checkpoint']);
        if 'text_log' in input_parameters.keys():
            text_log = int(input_parameters['text_log'])
        else:
            text_log = checkpoint;
        duration = input_parameters['duration'];
        time_step = fp.Variable(dt)
        
        
        eqn0 = fp.TransientTerm(coeff=1.,var=phi_p) == fp.DiffusionTerm(coeff=M_protein*FE.dmu_p_dphi_p(phi_p,phi_r),var=phi_p) + fp.DiffusionTerm(coeff=M_protein*FE.dmu_p_dphi_r(phi_p,phi_r),var=phi_r) - fp.DiffusionTerm(coeff=(M_protein,FE.kappa),var=phi_p) + M_protein*return_noise(mesh,time_step,noise_strength);

        if not int(input_parameters['mu_r']):
            eqn1 = fp.TransientTerm(coeff=1.,var=phi_r) == fp.DiffusionTerm(coeff=M_rna,var=phi_r) + rates_1.production(phi_p,phi_r, t) - rates_1.degradation(phi_p,phi_r) + M_rna*return_noise(mesh,time_step,noise_strength)
            #eqn1 = fp.TransientTerm(coeff=1.,var=phi_r_1) == fp.DiffusionTerm(coeff=M_rna,var=phi_r_1) + rates_1.production(phi_p,phi_r_1, t) - rates_1.degradation(phi_p,phi_r_1) + M_rna*return_noise(mesh,time_step,noise_strength)
           # eqn2 = fp.TransientTerm(coeff=1.,var=phi_r_2) == fp.DiffusionTerm(coeff=1.,var=phi_r_2) + rates_2.production(phi_p,phi_r_2, t) - rates_2.degradation(phi_p,phi_r_2) + M_rna*return_noise(mesh,time_step,noise_strength)
            
        else:
            eqn1 = fp.TransientTerm(coeff=1.,var=phi_r) == fp.DiffusionTerm(coeff=M_rna*FE.dmu_p_dphi_r(phi_p,phi_r),var=phi_p) + fp.DiffusionTerm(coeff=M_rna*FE.dmu_r_dphi_r(phi_p,phi_r),var=phi_r) + rates_1.production(phi_p,phi_r,t) - rates_1.degradation(phi_p,phi_r) +M_rna* return_noise(mesh,time_step,noise_strength)




        # ## Generate output directory strcuctures

        # In[9]:

        """
        Generates_1 overall output_structure
                Folder = 'Output/' +  output_folder_specified + 'simulation_params' + 'seed'

            stats file will contain
                step number, time, dt, Radius, Pmin, Pmax, Rmin,Rmax,Pavg,Ravg, f,t_sim
        """
        
        # calculate average kpx
        kpx_average = rates_1.kpx_total# / np.sum(mesh.cellVolumes)
        
        
        
        output_directory = '/data/hschede/Output/' + args.o +'/'
        traj_dir = 'L_' +  str(round(nx*dx,2)) + '_phi_p_0_'+str(phi_p_0) + '_phi_r_0_'+ str(phi_r_0) + '_chiPR_'+ str(chi) + '_k_production_'+ str(k_production) +'_k_degradation_'+ str(k_degradation) + '_d_' + str(dimension);
        traj_dir = traj_dir + '_a_' + str(a) + '_b_' + str(b)+ '_c_' + str(c);
        traj_dir = traj_dir + '_rhos_' + str(rho_s) + '_rhor_' + str(rho_r)+ '_kappa_' + str(kappa);
        traj_dir = traj_dir + '_ca_' + str(c_alpha) + '_cb_' + str(c_beta);
        traj_dir = traj_dir + '_param_' + str(par_name) + '_' + str(par);
        traj_dir = traj_dir + '_coord_' + str(coordinates) + '_std_' + str(std) + '_kpx_' + str(kpx_average)
        
        rand_dir_id = '/' + str(seed) + '/'
        output_dir = output_directory + traj_dir + rand_dir_id;
        os.makedirs(output_dir);
        os.makedirs(output_dir+ 'Images/');
        os.makedirs(output_dir+ 'Mesh/');
        print(output_dir)

        # Initialize file for writing stats
        with open(output_dir+ "/stats.txt", 'w+') as stats:
            stats.write("\t".join(["step", "t", "dt",'radius', 'area','distances', 'eccentricity','vacuole_num', 'vacuole_areas',
                                   "Pmin", "Pmax", 'Rmin','Rmax', 'Rtotal', 'Kp_step',
                                   "Pavg","Ravg", "f","t_sim"]) + "\n")

        write_input_params(output_dir + '/input_params.txt',input_parameters)
        

        ## Solve the Equation

        # To solve the equation a simple time stepping scheme is used which is decreased or increased based on whether the residual decreases or increases. A time step is recalculated if the required tolerance is not reached. In addition, the time step is kept under 1 unit. The data is saved out every 10 steps.

        elapsed = 0.0
        steps = 0
        dphip_dt = None

        phi_p.updateOld()
        phi_r.updateOld()
        #phi_r_1.updateOld()
        #phi_r_2.updateOld()
        #phi_r = phi_r_1 + phi_r_2
        
        # set up coordinates for interpolation
        coord_xy = np.array(list(zip(mesh.x, mesh.y)))
        X = np.linspace(min(coord_xy[:,0]), max(coord_xy[:,0]), nx * dx)
        Y = np.linspace(min(coord_xy[:,1]), max(coord_xy[:,1]), nx * dx)
        X, Y = np.meshgrid(X, Y)
        
            



        while (elapsed <= duration) and (steps <= total_steps) and (dt>dt_min):
            
#             print('phir_1')
#             print(max(phi_r_1))
#             print('\nphir_2')
#             print(max(phi_r_2))
            # Assert the approximations are valid
            assert max(phi_r) < 1, "Phi_r value surpassed 1.0. Aborting due to inaccurate approximations"
            
            # Recall equation to incorporate noise
#             eqn0 = fp.TransientTerm(coeff=1.,var=phi_p) == fp.DiffusionTerm(coeff=M_protein*FE.dmu_p_dphi_p(phi_p,phi_r),var=phi_p) + fp.DiffusionTerm(coeff=M_protein*FE.dmu_p_dphi_r(phi_p,phi_r),var=phi_r) - fp.DiffusionTerm(coeff=(M_protein,FE.kappa),var=phi_p) + M_protein*return_noise(mesh,time_step,noise_strength);

#             if not int(input_parameters['mu_r']):
#                 eqn1 = fp.TransientTerm(coeff=1.,var=phi_r) == fp.DiffusionTerm(coeff=M_rna,var=phi_r) + rates_1.production(phi_p,phi_r, t) - rates_1.degradation(phi_p,phi_r) + M_rna*return_noise(mesh,time_step,noise_strength)
#                 #eqn1 = fp.TransientTerm(coeff=1.,var=phi_r_1) == fp.DiffusionTerm(coeff=M_rna,var=phi_r_1) + rates_1.production(phi_p,phi_r_1, t) - rates_1.degradation(phi_p,phi_r_1) + M_rna*return_noise(mesh,time_step,noise_strength)
#                 #eqn2 = fp.TransientTerm(coeff=1.,var=phi_r_2) == fp.DiffusionTerm(coeff=1.,var=phi_r_2) + rates_2.production(phi_p,phi_r_2, t) - rates_2.degradation(phi_p,phi_r_2) + M_rna*return_noise(mesh,time_step,noise_strength)

#             else:
#                 eqn1 = fp.TransientTerm(coeff=1.,var=phi_r) == fp.DiffusionTerm(coeff=M_rna*FE.dmu_p_dphi_r(phi_p,phi_r),var=phi_p) + fp.DiffusionTerm(coeff=M_rna*FE.dmu_r_dphi_r(phi_p,phi_r),var=phi_r) + rates_1.production(phi_p,phi_r,t) - rates_1.degradation(phi_p,phi_r) +M_rna* return_noise(mesh,time_step,noise_strength)

            res1 = eqn1.sweep(dt=dt)
            #res2 = eqn2.sweep(dt=dt)
            res0 = eqn0.sweep(dt=dt)

            #phi_r = phi_r_1 + phi_r_2

            if max(res0,res1) > tolerance:

                # anything in this loop will only be executed every $checkpoint steps
                if (steps % checkpoint == 0):
                    if (changing_chi==1):
                        fp.TSVViewer(vars=[phi_p,phi_r,FE.chi_eff(phi_r,phi_p)]).plot(filename=output_dir +"Mesh/mesh_{step}.txt".format(step=steps))
                    elif (changing_chi==2):
                        fp.TSVViewer(vars=[phi_p,phi_r,FE.chi_eff(phi_r)]).plot(filename=output_dir +"Mesh/mesh_{step}.txt".format(step=steps))

                    else:
                        fp.TSVViewer(vars=[phi_p,phi_r]).plot(filename=output_dir +"Mesh/mesh_{step}.txt".format(step=steps))

                    if (dimension==2) and (plot_flag):

                        fig, ax  =plt.subplots()
                        cs = ax.tricontourf(
                            mesh.x.value,mesh.y.value,
                            phi_p.value,
                            cmap=plt.cm.get_cmap("Blues"),
                            levels=np.linspace(0,1.15*c_beta,256)
                        )
                        fig.colorbar(cs)
                        ax.set_title(phi_p.name)
                        fig.savefig(fname=output_dir +'Images/P_step_{step}.png'.format(step=steps),dpi=300,format='png')
    #                    fp.MatplotlibViewer(vars=phi_p,levels=np.linspace(0,1.0,10),cmap=plt.cm.get_cmap('Blues')).plot(filename=output_dir +'Images/A_step_{step}.png'.format(step=steps))
                        if input_parameters['svg_flag']:
                            for c in cs.collections:
                                c.set_edgecolor("face")
                            fig.savefig(fname=output_dir +'Images/P_step_{step}.svg'.format(step=steps),dpi=600,format='svg')
                        plt.close()


    #                    fp.MatplotlibViewer(vars=phi_r,datamin=0,datamax=0.35,cmap=plt.cm.get_cmap('PuRd')).plot(filename=output_dir +'Images/B_step_{step}.png'.format(step=steps))

                        fig, ax = plt.subplots()
                        cs = ax.tricontourf(mesh.x.value,mesh.y.value,
                                            phi_r.value,
                                            cmap=plt.cm.get_cmap("PuRd"),
                                            levels=np.linspace(0,0.2,256)
                                            #levels=np.linspace(0,2.5e-1+1.15*c_beta/(k_degradation+1e-9),256)
                                            #levels=np.linspace(0,2.5e-1+1.15*k_production*c_beta/(k_degradation+1e-9),256)
                                           )
                        fig.colorbar(cs)
                        ax.set_title(phi_r.name)
                        fig.savefig(fname=output_dir +'Images/R_step_{step}.png'.format(step=steps),dpi=300,format='png')
                        if input_parameters['svg_flag']:
                            for c in cs.collections:
                                c.set_edgecolor("face")
                            fig.savefig(fname=output_dir +'Images/R_step_{step}.svg'.format(step=steps),dpi=600,format='svg')
                        plt.close()


                        if (changing_chi):
                            fig, ax  =plt.subplots()
                            cs = ax.tricontourf(
                                mesh.x,mesh.y,
                                FE.chi_eff(phi_r,phi_p).value,
                                cmap=plt.cm.get_cmap("RdYlGn"),
                                levels=np.linspace(-FE.chi-1e-3,FE.chi+1e-3,256)
                            )
                            fig.colorbar(cs)
                            ax.set_title('$ \chi $')
                            fig.savefig(fname=output_dir +'Images/chi_step_{step}.png'.format(step=steps),dpi=300,format='png')
                            if input_parameters['svg_flag']:
                                for c in cs.collections:
                                    c.set_edgecolor("face")
                                fig.savefig(fname=output_dir +'Images/chi_step_{step}.svg'.format(step=steps),dpi=600,format='svg')
                            plt.close()

                if (steps % text_log ==0):
                    
           
                    # perform interpolation
                    coord_z = phi_p.value
                    img_source = griddata(coord_xy, coord_z, (X, Y), method='cubic')
                    
                    #########

                    # Calculate features and spatial variables                        
                        
                    area_, distances_, eccentricity, vacuole_num, vacuole_areas_ = sim_tools.retrieve_condensate_properties(
                        img_source, int(nx * dx), coordinates)
                    distances = ' '.join(str(x) for x in distances_)
                    vacuole_areas = ' '.join(str(x) for x in vacuole_areas_)
                    
                    # calculate chemical potentials
                    mup_ = FE.mu_p(phi_p=phi_p, phi_r=phi_r)
                    mur_ = FE.mu_r(phi_p=phi_p, phi_r=phi_r)
                    
                    # calculate gradients
                    grad_phip_ = np.expand_dims(np.array(phi_p.grad), 0)
                    grad_phir_ = np.expand_dims(np.array(phi_r.grad), 0)
                    
                    # calculate dphi/dt
                    dphip_dt_ = phi_p - phi_p.old
                    dphir_dt_ = phi_r - phi_r.old
                    
                    # calculate second derivatives
                    dmup_dphip_ = FE.dmu_p_dphi_p(phi_p,phi_r)
                    dmup_dphir_ = FE.dmu_p_dphi_r(phi_p,phi_r)
                    dmur_dphir_ = FE.dmu_r_dphi_r(phi_p,phi_r)
                    
                    # calculate divergence of mup
                    grad_mup_ = dmup_dphip_ * grad_phip_ + dmup_dphir_ * grad_phir_
                    
                    # calculate determinant of jacobian
                    det_J_ = (dmup_dphip_ * dmur_dphir_) - (dmup_dphir_ ** 2)
                    
                    # calculate free energy
                    free_energy_ = FE.f(phi_p, phi_r)
                    
                    
                    if type_ == 'gaussian':

                        # calculate the projection of grad_phip onto the vector connecting the RNA source and protein condensate     
                        grad_mup_x = np.array(grad_mup_[0][0]).flatten()
                        grad_mup_y = np.array(grad_mup_[0][1]).flatten()
                        c = np.vstack((grad_mup_x,grad_mup_y))

                        vector_projection = np.array(coordinates[0])
                        grad_mup_projection_ = np.dot(c.T, vector_projection) / np.linalg.norm(vector_projection)

                        grad_mup_projection_ = np.expand_dims(grad_mup_projection_, 0)
                        grad_mup_projection_magnitude_ = np.abs(grad_mup_projection_)
                    
                    
                  

                    if steps == 0:
                        grad_phip = grad_phip_
                        grad_phir = grad_phir_
                        grad_mup = grad_mup_
                        det_J = det_J_
                        dphip_dt = dphip_dt_
                        dphir_dt = dphir_dt_
                        mup = mup_
                        mur = mur_
                        dmup_dphip = dmup_dphip_
                        dmup_dphir = dmup_dphir_
                        dmur_dphir = dmur_dphir_
                        phir_stack = phi_r.value
                        phip_stack = phi_p.value
                        free_energy = free_energy_
                    
                        if type_ == 'gaussian':
                            grad_mup_projection = grad_mup_projection_
                            grad_mup_projection_magnitude = grad_mup_projection_magnitude_
                        
                    else:
                        grad_phip = np.vstack([grad_phip, grad_phip_])
                        dphip_dt = np.vstack([dphip_dt, dphip_dt_])
                        dphir_dt = np.vstack([dphir_dt, dphir_dt_])
                        grad_phir = np.vstack([grad_phir, grad_phir_])
                        grad_mup = np.vstack([grad_mup, grad_mup_])
                        det_J = np.vstack([det_J, det_J_])
                        mur = np.vstack([mur, mur_])
                        mup = np.vstack([mup, mup_])
                        dmup_dphip = np.vstack([dmup_dphip, dmup_dphip_])
                        dmup_dphir = np.vstack([dmup_dphir, dmup_dphir_])
                        dmur_dphir = np.vstack([dmur_dphir, dmur_dphir_])
                        phir_stack = np.vstack([phir_stack, phi_r.value])
                        phip_stack = np.vstack([phip_stack, phi_p.value])
                        free_energy = np.vstack([free_energy, free_energy_])
                        
                        if type_ == 'gaussian':
                            grad_mup_projection = np.vstack([grad_mup_projection, grad_mup_projection_])
                            grad_mup_projection_magnitude = np.vstack([grad_mup_projection_magnitude, grad_mup_projection_magnitude_])

                        
                        
                    with open(output_dir+ "/stats.txt", 'a') as stats:
                        stats.write("\t".join([str(it) for it in [
                            steps,
                            t.value,
                            dt, 
                            sim_tools.get_radius(phi_p,mesh,dimension=dimension,threshold=0.5*(c_alpha+c_beta)),
                            area_,
                            distances,
                            eccentricity,
                            vacuole_num,
                            vacuole_areas,
                            min(phi_p), max(phi_p), min(phi_r),
                            max(phi_r),
                            np.array(phi_r * mesh.cellVolumes).sum(),
                            rates_1.rate,
                            np.mean(phi_p*mesh.cellVolumes) / np.mean(mesh.cellVolumes),
                            np.mean(phi_r*mesh.cellVolumes) / np.mean(mesh.cellVolumes),
                            np.sum((FE.f(phi_p,phi_r)*mesh.cellVolumes).value),
                            str(round((timeit.default_timer()-start),2))]]) + "\n")

                steps += 1
                elapsed += dt
                t.value = t.value +dt
                
                dt *= 1.1
                dt = min(dt, dt_max)
                time_step.value = dt;
                phi_p.updateOld()
                try:
                    phi_r.updateOld()
                except:
                    phi_r_1.updateOld()
                    phi_r_2.updateOld()

            else:
                dt *= 0.8
                time_step.value = dt;
                phi_p[:] = phi_p.old
                phi_r[:] = phi_r.old
                
             
        # Save hdf5 files
        f = h5py.File(os.path.join(output_dir + 'spatial_variables.hdf5'),'w')
        dset1 = f.create_dataset("grad_phip", data=grad_phip)
        dset2 = f.create_dataset("grad_phir", data=grad_phir)
        dset3 = f.create_dataset("grad_mup", data=grad_mup)
        dset4 = f.create_dataset("det_J", data=det_J)
        dset5 = f.create_dataset("dphip_dt", data=dphip_dt)
        dset5 = f.create_dataset("dphir_dt", data=dphir_dt)
        dset6 = f.create_dataset("mup", data=mup)
        dset7 = f.create_dataset("mur", data=mur)
        dset8 = f.create_dataset("dmup_dphip", data=dmup_dphip)
        dset9 = f.create_dataset("dmup_dphir", data=dmup_dphir)
        dset10 = f.create_dataset("dmur_dphir", data=dmur_dphir)
        dset11 = f.create_dataset("phi_r", data=phir_stack)
        dset12 = f.create_dataset("phi_p", data=phip_stack)
        dset12 = f.create_dataset("free_energy", data=free_energy)
        dset13 = f.create_dataset("kpx", data=rates_1.kpx)

        if type_ == 'gaussian':
            dset6 = f.create_dataset("grad_mup_projection", data=grad_mup_projection)
            dset7 = f.create_dataset("grad_mup_projection_magnitude", data=grad_mup_projection_magnitude)
        
        f.close()
        
        # Save movies
        
        # movies
        if type_ == 'gaussian':
            list_matrices = ['phi_r','phi_p', 'dphip_dt', 'dphir_dt', 'grad_mup',
                             'grad_phip', 'grad_phir', 'det_J',
                             'grad_mup_projection','grad_mup_projection_magnitude',
                             'mup', 'mur', 'dmup_dphip', 'dmup_dphir', 'dmur_dphir', 'free_energy'
                            ]
        else:
            list_matrices = ['phi_r', 'phi_p', 'dphip_dt', 'dphir_dt', 'grad_mup', 'grad_phip',
                             'grad_phir', 'det_J', 'mup', 'mur','dmup_dphip',
                             'dmup_dphir', 'dmur_dphir', 'free_energy'
                            ]
        
        write_movie_from_hdf5(output_dir, list_matrices, mesh)
        
        # gifs
        try:
            if dimension==2 and (plot_flag):
                save_movie(output_dir +'Images/',duration=0.25)
                if input_parameters['svg_flag']:
                    bash_cmd = 'rm '+ output_dir +'Images/*.png'
                    res = subprocess.check_output(['bash','-c',bash_cmd])

            elif dimension==3 and (plot_flag):
                generate_images_3D(output_dir,label_idx=3,N=nx,colormap="Blues",vmin=0.0,vmax=1.0,opacity=0.2)
                generate_images_3D(output_dir,label_idx=4,N=nx,colormap="PuRd",vmin=0.0,vmax=0.35,opacity=0.2)
                save_movie(output_dir +'Images/',duration=0.25)
        except:
            pass




if __name__ == "__main__":
    """
        Function is called when python code is run on command line and calls run_CH
        to initialize the simulation
    """
    parser = argparse.ArgumentParser(description='Take output filename to run CH simulations')
    parser.add_argument('--i',help="Name of input params", required = True);
    parser.add_argument('--p',help="Name of parameter file", required = False);
    parser.add_argument('--pN',help="Parameter number from file (indexed from 1)", required = False);

    parser.add_argument('--o',help="Name of output folder", required = True);
    args = parser.parse_args();

    run_CH(args);
