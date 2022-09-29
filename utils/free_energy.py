#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modules that are used to define the:

    1. Underlying free-energy landscape

        -   Double-well potential
        -   Flory-Huggins potential

    2. RNA synthesis and degradation reactions, including as a function of space
"""

import numpy as np

def gaussian(std, x, min_threshold=0.00):
    value = 1 / (2*np.pi*std) * np.exp(-.5 * ((x/std) ** 2))
    if value  < min_threshold: # if the value is under a threshold, reset it
        value = min_threshold
    return value

def exponential(lambda_, x, min_threshold=0.00):
    value = lambda_ * np.exp(-lambda_ * x)
    if value  < min_threshold: # if the value is under a threshold, reset it
        value = min_threshold
    return value

def retrieve_spatial_weights(mesh, coordinates, stds, distribution, scale=True, dimension=2):
    """Return a normalized vector representing sites of RNA production. specified by the size of the mesh, 
    the specified coordinate and standard deviation for the coordinate spot"""
    mesh_coordinates = mesh.cellCenters.value.T
    #distance_total = np.zeros((mesh.shape[0] * mesh.shape[1]))
    distance_total = []
    
    for idx, coordinate in enumerate(coordinates):
        distances = []
        # iterate over coordinates in mesh
        for mesh_coord in mesh_coordinates:
            x_dist = np.abs(coordinate[0] - mesh_coord[0])
            y_dist = np.abs(coordinate[1] - mesh_coord[1])
            vector = [x_dist, y_dist]  
            if dimension == 3:
                z_dist = np.abs(coordinate[2] - mesh_coord[2])
                vector = [x_dist, y_dist, z_dist]  
            #vector = np.sqrt(x_dist ** 2 + y_dist **2) # calculate distance vector
            norm = np.linalg.norm(vector) # calculate norm
            if distribution == 0:
                g_norm = gaussian(stds[idx], norm) # from gaussian distribution
            else:
                g_norm = exponential(stds[idx], norm)

            distances.append(g_norm)
        distances = np.array(distances) / np.max(distances) # scale so that max level is 1
        # if we want the different gaussians to have the same area then we 
        # must find the constant such that the sum is one
        if scale:
            # needs to take into account the size of the cellvolumes
            scale_factor = 1 / np.sum(distances * mesh.cellVolumes)
            distances = distances * scale_factor
        distance_total.append(distances)
    return distance_total



class RNA_reactions():
    """
        RNA reactions allow for definition of production and degradation reactions.

        **Class initialization variables**

        -   k_production = rate of RNA synthesis (kp),
        -   k_degradation = rate of RNA degradation (kd),
        -   threshold = protein concentration above which RNA synthesis is catalyzed (default = 0.0, thr)
        -   t_change = simulation time after which k_production is increased by multiplier (default = 0.0)
        -   multiplier = rate at which k_production is increased after t_change (default = 1.0, mul)
        -   m1  =   order of protein concentration in RNA production (default =1.0)
        -   kp_noise    =   STD of white noise added to on_rate (default = 0.0)
        -   mesh = the grid that is required for generating the spatial profile of RNA production
        -   coordinates = list of tuplies indicating the desired x,y positions of the individual centroids of RNA production
        -   std = list of floats indicating the standard deviations of the individual gaussians
    """
    def __init__(self,k_production,k_degradation, mesh, coordinates, std, distribution,
                 threshold=0.0,t_change=0.0,multiplier=1.0,m1=1.0,kp_noise=0.0, scale=True, dimension=2):
        self.k_production = k_production;
        self.rate = k_production # initialize the random rate with the true k_production
        self.k_degradation= k_degradation;
        if isinstance(coordinates, float):
            self.type = 'gradient'
        else:
            self.type = False
            try:
                print(mesh.cellVolumes.shape)
                self.kpx_total = np.sum(mesh.cellVolumes)#np.sum(np.ones(mesh.shape).flatten() * mesh.cellVolumes)
            except AttributeError:
                self.kpx_total = np.sum(np.ones(mesh.dim).flatten() * mesh.cellVolumes)
            #self.kpx_total *= self.k_production
            
            
        # Calculate the distribution of spatial RNA production
        if isinstance(coordinates, list):
            if len(coordinates) != 0:
                self.location = retrieve_spatial_weights(mesh, coordinates, std, distribution, scale=scale, dimension=dimension)
                # Calculate the sum of k_production over the mesh          
                self.kpx_total = 0.
                for idx_, entry in enumerate(self.location):
                    try:
                        self.kpx_total += np.sum(entry * mesh.cellVolumes) * self.k_production[idx_]
                    except:
                        self.kpx_total += np.sum(entry * mesh.cellVolumes) 
                if isinstance(self.k_production, float):
                    self.kpx_total *= self.k_production
            else:
                self.location = False
		self.kpx_total *= self.k_production
        else:
            self.location = False
            self.kpx_total *= self.k_production

        # Gradient is constant across an axis=0
        if isinstance(coordinates, float):
            mesh_normalized = mesh.cellCenters.value + np.abs(mesh.cellCenters.value.min())
            mesh_normalized /= np.max(mesh_normalized)           
            self.location = mesh_normalized[0,:] ** coordinates # change steepness of gradient
            self.location *= k_production
            
            # Calculate the sum of k_production over the mesh           
            self.kpx_total = np.sum(self.location * mesh.cellVolumes)
            
            
            
        self.threshold = threshold;
        self.t_change = t_change;
        self.multiplier = multiplier;
        self.m1 = m1;
        self.kp_noise =kp_noise;
        
        
        if not isinstance(self.location, bool): # If coordinates were provided
                    if self.type == 'gradient':
                        
                        self.kpx = k_production * self.location
                    else:
                        added_terms = np.zeros(self.location[0].shape)

                        # try if kp is a list
                        try:
                            for entry in range(len(self.location)):
                                added_terms += self.location[entry] * self.k_production[entry]

                            self.kpx = added_terms
                        
                        # except if kp is not a list
                        except:
                            for entry in range(len(self.location)):
                                added_terms += self.location[entry]
                            self.kpx = k_production * added_terms
        
    def production(self,phi_p,phi_r, t):
        """
        Computes the first-order protein-dependent RNA synthesis rate as follows:

        .. math::
            rate = k_{p}(\phi_p>=thr)(\phi_p)(1 + (mul-1)(t>=t_{change}))

        - Takes in phi_p, phi_r, and simulation time t
        - Computes and returns rate of production
        """
        
        
        if self.kp_noise > 0:
            rate = max(0,self.k_production+np.random.normal(scale=self.kp_noise)) # added noise
        else:
            rate = self.k_production

        self.rate = rate

        if not isinstance(self.location, bool): # If coordinates were provided
            if self.type == 'gradient':
                p = self.kpx *((phi_p-self.threshold)**self.m1)*(phi_p>self.threshold)*(1 + (self.multiplier-1)*(t>=self.t_change))
            
            else:

                try:
                    p = ((phi_p - self.threshold) ** self.m1) * (phi_p > self.threshold) * (1 + (self.multiplier - 1) * (t >= self.t_change)) * self.kpx
                    
                except:
                    p = ((phi_p-self.threshold)**self.m1)*(phi_p>self.threshold)*(1 + (self.multiplier-1)*(t>=self.t_change)) * self.kpx
                    
                
        else: # No spatial restriction
            p = (rate)*((phi_p-self.threshold)**self.m1)*(phi_p>self.threshold)*(1 + (self.multiplier-1)*(t>=self.t_change))
            
        return(p)

    def degradation(self,phi_p,phi_r):
        """
        Computes the first-order RNA degradation as follows:

        .. math::
            rate = k_{d}\phi_{r}

        - Takes in phi_p, phi_r
        - Returns rate of RNA degradation
        """
        return(self.k_degradation*phi_r)



"""
    Free energies are defined to return f, chemical potential and their first derivatives
"""

class free_energy_changing_chi_LG():
    """
    LG free-energy defines a free-energy for RNA and protein with the following features:

        -   Double-well potential for protein
        -   Second-order term for RNA
        -   Non-linear :math:`\chi_{0}` for RNA-protein interactions as defined below:

        .. math::
            \chi_{0} = -\chi \phi_{p} \phi_{r} + a \phi_{p} \phi^{2}_{r} = (\chi_{eff})\phi_{p}\phi_{r}

    **Input variables**

    To define a variable of this class, the following parameters are required:

        -   c_alpha =   Dilute coexistence protein concentration
        -   c_beta  =   Dense coexistence protein concentration
        -   rho_s   =   Height of double-well potential
        -   rho_r   =   Coefficient of 2nd order term for RNA
        -   chi     =   2-nd order interaction term coefficient (>0)
        -   a       =   3-nd order interaction term coefficient (>0)
        -   kappa   =   Surface tension term
    """
    def __init__(self,c_alpha,c_beta,rho_s,rho_r,chi,kappa,a=0.0,b=0.0,c=0.0):
        self.c_alpha = c_alpha;
        self.c_beta = c_beta;
        self.rho_s = rho_s;
        self.rho_r = rho_r;
        self.chi = chi;
        self.kappa = kappa;
        self.a = a;
        self.b = b;
        self.c = c;

    # homogeneous bulk free energy
    def chi_eff(self,phi_r,phi_p):
        """
        Returns :math:`\chi_{eff} (\phi_r ,\phi_p)`

        .. math::
            \chi_{eff} = -\chi + a \phi_{r} + b \phi_{p} + c \phi_r \phi_p
        """
        return (-self.chi + self.a*phi_r + self.b*phi_p + self.c*phi_r*phi_p)

    def f_0(self,phi_p,phi_r):
        """
        Returns bulk free-energy of the form:

        .. math::
            f_{0} = \\rho_{s} ( \phi_{p} - c_{\\alpha})^{2}(\phi_{p} - c_{\\beta})^{2} + \\rho_{r} \phi^{2}_{r} + \chi_{0}(\phi_{p},\phi_{r})

        """
        return self.rho_s * (phi_p - self.c_alpha)**2 * (self.c_beta-phi_p)**2 + self.chi_eff(phi_r,phi_p) * phi_p * phi_r + self.rho_r * (phi_r)**2

    # free energy
    def f(self,phi_p,phi_r):
        """
        Returns overall free-energy including gradient (surface-tension) terms:

        .. math::
            f = f_{0} + \\kappa(\\nabla \phi_p)^{2}
        """
        return self.f_0(phi_p,phi_r)+ .5*self.kappa*(phi_p.grad.mag)**2

    # chemical potential of protein
    def mu_p(self,phi_p,phi_r):
        """
        Returns protein chemical potential

        .. math::
            \mu_{p} = \\frac{df}{d \phi_{p}}
        """
        return 2* self.rho_s * (self.c_alpha-phi_p) * (self.c_beta-phi_p) * (self.c_alpha + self.c_beta - 2*phi_p) -self.chi*phi_r + self.a*phi_r*phi_r  + 2*self.b*phi_p*phi_r + 2*self.c*phi_r*phi_r*phi_p 

    # chemical potential of RNA
    def mu_r(self,phi_r,phi_p):
        """
        Returns RNA chemical potential

        .. math::
            \mu_{r} = \\frac{df}{d \phi_{r}}
        """
        return(2*self.rho_r*phi_r -self.chi*phi_p + 2*self.a*phi_p*phi_r  + self.b*phi_p*phi_p + 2*self.c*phi_r*phi_p*phi_p  );

    # first derivatives of chemical potential

    def dmu_p_dphi_p(self,phi_p,phi_r):
        """
        Returns derivative of protein chemical potential with protein concentration

        .. math::
             \\frac{d^{2}f}{d \phi_{p}^{2}}
        """
        if self.c_alpha==0.0 and (self.c_beta==0.0):
            Jpp = 2 * self.rho_s + self.b*2*phi_r +self.c*2*phi_r*phi_r;
        
        else:
            
            Jpp = 2 * self.rho_s * ((self.c_alpha - phi_p)**2 + 4*(self.c_alpha - phi_p)*(self.c_beta - phi_p) + (self.c_beta - phi_p)**2) + self.b*2*phi_r +self.c*2*phi_r*phi_r;
        
        return Jpp

    def dmu_p_dphi_r(self,phi_p,phi_r):
        """
        Returns mixed second derivative of free-energy

        .. math::
             \\frac{d^{2}f}{d \phi_{p} \phi_{r}}
        """
        return (-self.chi + 2*self.a*phi_r + 2*self.b*phi_p + 4*self.c*phi_p*phi_r);

    def dmu_r_dphi_r(self,phi_p,phi_r):
        """
        Returns derivative of RNA chemical potential with RNA concentration

        .. math::
             \\frac{d^{2}f}{d \phi_{r}^{2}}
        """
        return (2*self.rho_r + 2*self.a*phi_p + 2*self.c*phi_p*phi_p) ;



class free_energy_changing_chi():
    """
    Free-energy defines a free-energy for RNA and protein with the following features:

        -   Double-well potential for protein
        -   Second-order term for RNA
        -   Non-linear :math:`\chi_{0}` that is phenomenologically derived:

        .. math::
            \chi_{eff} = \chi_{0} = \chi (1 - p e^{(\phi_{r} - \phi_{r^{*}})^{2}/(2a^{2}) } )

    **Input variables**

    To define a variable of this class, the following parameters are required:

        -   c_alpha =   Dilute coexistence protein concentration
        -   c_beta  =   Dense coexistence protein concentration
        -   rho_s   =   Height of double-well potential
        -   rho_r   =   Coefficient of 2nd order term for RNA
        -   kappa   =   Surface tension term
        -   chi     =   2-nd order interaction term coefficient (>0)
        -   a       =   width of gaussian around charge-balance concentration
        -   ratio   =   charge-balance concentration ratio of c_beta
        -   p       =   depth of attractive :math:`\chi_{eff}` well which is :math:`\chi (1-p)`
    """
    def __init__(self,c_alpha,c_beta,rho_s,rho_r,chi,kappa,a=0.03,ratio=5,p=1.5):
        self.c_alpha = c_alpha;
        self.c_beta = c_beta;
        self.rho_s = rho_s;
        self.rho_r = rho_r;
        self.chi = chi;
        self.kappa = kappa;
        self.a = a;
        self.phi_r_star = c_beta/ratio;
        self.p = p;

    # homogeneous bulk free energy
    def chi_eff(self,phi_r):
        """
        Returns :math:`\chi_{eff} (\phi_r)`

        .. math::
            \chi_{eff} = \chi_{0} = \chi (1 - p e^{(\phi_{r} - \phi_{r^{*}})^{2}/(2a^{2}) } )
        """
        return (self.chi*(1 - self.p*np.exp(-(phi_r - self.phi_r_star)**2/(2*self.a**2))))

    def f_0(self,phi_p,phi_r):
        """
        Returns bulk free-energy of the form:

        .. math::
            f_{0} = \\rho_{s} ( \phi_{p} - c_{\\alpha})^{2}(\phi_{p} - c_{\\beta})^{2} + \\rho_{r} \phi^{2}_{r} + \chi_{0}(\phi_{p},\phi_{r})

        """
        return self.rho_s * (phi_p - self.c_alpha)**2 * (self.c_beta-phi_p)**2 + self.chi_eff(phi_r) * phi_p * phi_r + self.rho_r * (phi_r)**2

    # free energy
    def f(self,phi_p,phi_r):
        """
        Returns overall free-energy including gradient (surface-tension) terms:

        .. math::
            f = f_{0} + \\kappa(\\nabla \phi_p)^{2}
        """
        return self.f_0(phi_p,phi_r)+ .5*self.kappa*(phi_p.grad.mag)**2

    # chemical potential
    def mu_p(self,phi_p,phi_r):
        """
        Returns protein chemical potential

        .. math::
            \mu_{p} = \\frac{df}{d \phi_{p}}
        """
        return 2* self.rho_s * (self.c_alpha-phi_p) * (self.c_beta-phi_p) * (self.c_alpha + self.c_beta - 2*phi_p) +self.chi_eff(phi_r) * phi_r

    # chemical potential of RNA
    def mu_r(self,phi_r,phi_p):
        """
        Returns RNA chemical potential

        .. math::
            \mu_{r} = \\frac{df}{d \phi_{r}}
        """
        return(2*self.rho_r*phi_r +self.chi_eff(phi_r)*phi_p + - (self.chi_eff(phi_r)-self.chi)*(phi_r-self.phi_r_star)/(self.a**2)*phi_p*phi_r);

    # first derivatives of chemical potential
    def dmu_p_dphi_p(self,phi_p,phi_r):
        """
        Returns derivative of protein chemical potential with protein concentration

        .. math::
             \\frac{d^{2}f}{d \phi_{p}^{2}}
        """
        return 2 * self.rho_s * ((self.c_alpha - phi_p)**2 + 4*(self.c_alpha - phi_p)*(self.c_beta - phi_p) + (self.c_beta - phi_p)**2)

    def dmu_p_dphi_r(self,phi_p,phi_r):
        """
        Returns mixed second derivative of free-energy

        .. math::
             \\frac{d^{2}f}{d \phi_{p} \phi_{r}}
        """
        return self.chi_eff(phi_r) - (self.chi_eff(phi_r)-self.chi)*phi_r*(phi_r-self.phi_r_star)/(self.a**2) ;

    def dmu_r_dphi_r(self,phi_p,phi_r):
        """
        Returns derivative of RNA chemical potential with RNA concentration

        .. math::
             \\frac{d^{2}f}{d \phi_{r}^{2}}
        """
        return (2*self.rho_r - 2*(self.chi_eff(phi_r)-self.chi)*(phi_r-self.phi_r_star)/(self.a**2)*phi_p + 2*(self.chi_eff(phi_r)-self.chi)*((phi_r-self.phi_r_star)**2-(self.a**2))/(self.a**4)*phi_p*phi_r) ;


class free_energy_FH_changing_chi_LG():
    """
    Flory-Huggins(FH) free-energy defines a free-energy for RNA and protein with the following features:

        -   Protein-solvent attractions promote phase separation in absence of RNA
        -   Non-linear :math:`\chi_{0}` for RNA-protein interactions as defined below:

        .. math::
            \chi_{0} = -\chi \phi_{p} \phi_{r} + a \phi_{p} \phi^{2}_{r} = (\chi_{eff})\phi_{p}\phi_{r}

    **Input variables**

    To define a variable of this class, the following parameters are required:

        -   c_alpha =   Dilute coexistence protein concentration (symmetric about 0.5)
        -   c_beta  =   Dense coexistence protein concentration (symmetric about 0.5)
        -   rho_s   =   Height of double-well potential (not required- needs to be eliminated)
        -   chi     =   2-nd order interaction term coefficient (>0)
        -   a       =   3-rd order interaction term coefficient (>0)
        -   b       =   3-rd order interaction term (>0)
        -   c       =   4th order term (>0)
        -   kappa   =   Surface tension term
        -   chi_ps  =   Interaction b/w protein and solvent
        -   chi_rs  =   Interaction b/w RNA and solvent
        -   r       =   Polymerization of RNA/protein (>=1.0)

    Note: The strength of protein-solvent interactions for :math:`\chi_{ps}` is inferred from the
    single-component phase separation condition derived from binodal symmetric about 0.5 & having
    no polymerization factors in entropic terms as follows:

    .. math::
        \chi_{ps}   =   \\frac{log( \\frac{c_{\\beta}}{1 - c_{\\beta}} )}{2 c_{\\beta} - 1}

    Similarly, the RNA solvent interactions are assumed to be :math:`\chi_{rs} = 0`
    """
    def __init__(self,c_alpha,c_beta,rho_s,rho_r,chi,kappa,a,b,c,chi_rs=0,chi_ps=None,r=1.0):
        self.c_alpha = c_alpha;
        self.c_beta = c_beta;
        self.rho_s = rho_s;
        self.chi = chi;
        self.kappa = kappa;
        if chi_ps is None:
            self.chi_ps = np.log(c_beta/(1-c_beta))/(2*c_beta-1);
        else:
            self.chi_ps = chi_ps
        self.chi_rs = chi_rs;
        print(self.chi_ps)
        self.a = a;
        self.b = b;
        self.c = c;
        self.r = r;

    def chi_eff(self,phi_r,phi_p):
        """
        Returns :math:`\chi_{eff} (\phi_r)`

        .. math::
            \chi_{eff} = -\chi + a \phi_{r} + b \phi_{p} + c \phi_r \phi_p
        """
        return (-self.chi + self.a*phi_r + self.b*phi_p + self.c*phi_r*phi_p)

    # homogeneous bulk free energy
    def f_0(self,phi_p,phi_r):
        """
        Returns bulk free-energy of the form:

        .. math::
            f_{0} = \sum_{i} \phi_{i} log( \phi_{i}) + \sum_{i,j; j>i} \chi_{ij} \phi_{i} \phi_{j}

        where :math:`\chi_{pr} = \chi_{eff}`
        """
        return (phi_p)*np.log(phi_p)/self.r + phi_r*np.log(phi_r)/self.r + (1-phi_p-phi_r)*np.log((1-phi_p-phi_r)) + self.chi_eff(phi_r,phi_p)*phi_p*phi_r + self.chi_ps*phi_p*(1-phi_p-phi_r) + self.chi_rs*phi_r*(1-phi_p-phi_r)

    # free energy
    def f(self,phi_p,phi_r):
        """
        Returns overall free-energy including gradient (surface-tension) terms:

        .. math::
            f = f_{0} + \\kappa(\\nabla \phi_p)^{2}
        """
        return self.f_0(phi_p,phi_r)+ .5*self.kappa*(phi_p.grad.mag)**2

    # chemical potential
    def mu_p(self,phi_p,phi_r):
        """
        Returns protein chemical potential

        .. math::
            \mu_{p} = \\frac{df}{d \phi_{p}}
        """
        return ((1/self.r -1) + (np.log(phi_p)/self.r - np.log(1-phi_p-phi_r)) -self.chi*phi_r + self.a*phi_r*phi_r  + 2*self.b*phi_p*phi_r + 2*self.c*phi_r*phi_r*phi_p + self.chi_ps*(1-2*phi_p-phi_r) - self.chi_rs*phi_r)

    def mu_r(self,phi_p,phi_r):
        """
        Returns RNA chemical potential

        .. math::
            \mu_{r} = \\frac{df}{d \phi_{r}}
        """
        return ((1/self.r -1) + (np.log(phi_r)/self.r - np.log(1-phi_p-phi_r)) + self.chi_rs*(1-2*phi_r-phi_p) -self.chi_ps*phi_p -self.chi*phi_p + 2*self.a*phi_p*phi_r  + self.b*phi_p*phi_p + 2*self.c*phi_r*phi_p*phi_p) 

    # first derivative of chemical potential
    def dmu_p_dphi_p(self,phi_p,phi_r):
        """
        Returns derivative of protein chemical potential with protein concentration

        .. math::
             \\frac{d^{2}f}{d \phi_{p}^{2}}
        """
        return (1/(self.r*phi_p) + 1/(1-phi_p-phi_r) - 2*self.chi_ps + self.b*2*phi_r +self.c*2*phi_r*phi_r)

    def dmu_p_dphi_r(self,phi_p,phi_r):
        """
        Returns mixed second derivative of free-energy

        .. math::
             \\frac{d^{2}f}{d \phi_{p} \phi_{r}}
        """
        return (1/(1-phi_p-phi_r) - self.chi_ps -self.chi_rs -self.chi + 2*self.a*phi_r + 2*self.b*phi_p + 4*self.c*phi_p*phi_r);

    def dmu_r_dphi_r(self,phi_p,phi_r):
        """
        Returns derivative of RNA chemical potential with RNA concentration

        .. math::
             \\frac{d^{2}f}{d \phi_{r}^{2}}
        """
        return (1/(self.r*phi_r) + 1/(1-phi_p-phi_r)  - 2*self.chi_rs + 2*self.a*phi_p + 2*self.c*phi_p*phi_p);


class free_energy_FH_changing_chi():
    """
    Flory-Huggins(FH) free-energy defines a free-energy for RNA and protein with the following features:

        -   Protein-solvent attractions promote phase separation in absence of RNA
        -   Non-linear :math:`\chi_{0}` for RNA-protein interactions as defined below:

        .. math::
            \chi_{0} = \chi (1 - p e^{(\phi_{r} - \phi_{r^{*}})^{2}/(2a^{2}) }

    **Input variables**

    To define a variable of this class, the following parameters are required:

        -   c_alpha =   Dilute coexistence protein concentration (symmetric about 0.5)
        -   c_beta  =   Dense coexistence protein concentration (symmetric about 0.5)
        -   rho_s   =   Height of double-well potential (not required- needs to be eliminated)
        -   chi     =   2-nd order interaction term coefficient (>0)
        -   a       =   3-nd order interaction term coefficient (>0)
        -   kappa   =   Surface tension term

    Note: The strength of protein-solvent interactions for :math:`\chi_{ps}` is inferred from the
    single-component phase separation condition derived from binodal symmetric about 0.5 & having
    no polymerization factors in entropic terms as follows:

    .. math::
        \chi_{ps}   =   \\frac{log( \\frac{c_{\\beta}}{1 - c_{\\beta}} )}{2 c_{\\beta} - 1}

    Similarly, the RNA solvent interactions are assumed to be :math:`\chi_{rs} = 0`
    """
    def __init__(self,c_alpha,c_beta,rho_s,chi,kappa,a=0.03,ratio=5,p=1.5):
        self.c_alpha = c_alpha;
        self.c_beta = c_beta;
        self.rho_s = rho_s;
        self.chi = chi;
        self.kappa = kappa;
        self.chi_ps = np.log(c_beta/(1-c_beta))/(2*c_beta-1);
        self.chi_rs = 0;
        self.a = a;
        self.phi_r_star = c_beta/ratio;
        self.p = p;

    def chi_eff(self,phi_r):
        """
        Returns :math:`\chi_{eff} (\phi_r)`

        .. math::
            \chi_{eff} = \chi_{0} = \chi (1 - p e^{(\phi_{r} - \phi_{r^{*}})^{2}/(2a^{2}) } )
        """
        return (self.chi*(1 - self.p*np.exp(-(phi_r - self.phi_r_star)**2/(2*self.a**2))))

    # homogeneous bulk free energy
    def f_0(self,phi_p,phi_r):
        """
        Returns bulk free-energy of the form:

        .. math::
            f_{0} = \sum_{i} \phi_{i} log( \phi_{i}) + \sum_{i,j; j>i} \chi_{ij} \phi_{i} \phi_{j}

        where :math:`\chi_{pr} = \chi_{eff}`
        """
        return (phi_p)*np.log(phi_p) + phi_r*np.log(phi_r) + (1-phi_p-phi_r)*np.log((1-phi_p-phi_r)) + self.chi_eff(phi_r)*phi_p*phi_r + self.chi_ps*phi_p*(1-phi_p-phi_r) + self.chi_rs*phi_r*(1-phi_p-phi_r)

    # free energy
    def f(self,phi_p,phi_r):
        """
        Returns overall free-energy including gradient (surface-tension) terms:

        .. math::
            f = f_{0} + \\kappa(\\nabla \phi_p)^{2}
        """
        return self.f_0(phi_p,phi_r)+ .5*self.kappa*(phi_p.grad.mag)**2

    # chemical potential
    def mu_p(self,phi_p,phi_r):
        """
        Returns protein chemical potential

        .. math::
            \mu_{p} = \\frac{df}{d \phi_{p}}
        """
        return (np.log(phi_p/(1-phi_p-phi_r)) + self.chi_eff(phi_r)*phi_r + self.chi_ps*(1-2*phi_p-phi_r) - self.chi_rs*phi_r)

    def mu_r(self,phi_p,phi_r):
        """
        Returns RNA chemical potential

        .. math::
            \mu_{r} = \\frac{df}{d \phi_{r}}
        """
        return (np.log(phi_r/(1-phi_p-phi_r)) + self.chi_eff(phi_r)*phi_p + self.chi_rs*(1-2*phi_r-phi_p) -self.chi_ps*phi_p) - (self.chi_eff(phi_r)-self.chi)*(phi_r-self.phi_r_star)/(self.a**2)*phi_p*phi_r

    # first derivative of chemical potential
    def dmu_p_dphi_p(self,phi_p,phi_r):
        """
        Returns derivative of protein chemical potential with protein concentration

        .. math::
             \\frac{d^{2}f}{d \phi_{p}^{2}}
        """
        return (1/phi_p + 1/(1-phi_p-phi_r) - 2*self.chi_ps)

    def dmu_p_dphi_r(self,phi_p,phi_r):
        """
        Returns mixed second derivative of free-energy

        .. math::
             \\frac{d^{2}f}{d \phi_{p} \phi_{r}}
        """
        return (1/(1-phi_p-phi_r) + self.chi_eff(phi_r)  - self.chi_ps -self.chi_rs - (self.chi_eff(phi_r)-self.chi)*(phi_r-self.phi_r_star)/(self.a**2)*phi_r);

    def dmu_r_dphi_r(self,phi_p,phi_r):
        """
        Returns derivative of RNA chemical potential with RNA concentration

        .. math::
             \\frac{d^{2}f}{d \phi_{r}^{2}}
        """
        return (1/phi_r + 1/(1-phi_p-phi_r)  - 2*self.chi_rs - 2*(self.chi_eff(phi_r)-self.chi)*(phi_r-self.phi_r_star)/(self.a**2)*phi_p + 2*(self.chi_eff(phi_r)-self.chi)*((phi_r-self.phi_r_star)**2-(self.a**2))/(self.a**4)*phi_p*phi_r)


"""
    Simple free-energies with constant chi - backwards compatibility
"""
class free_energy():
    """
    Simple double-well free-energy with constant chi - provided for backwards compatibility

    Use free_energy_changing_chi_LG with a=0 (same result)
    """
    def __init__(self,c_alpha,c_beta,rho_s,chi,kappa):
        self.c_alpha = c_alpha;
        self.c_beta = c_beta;
        self.rho_s = rho_s;
        self.chi = chi;
        self.kappa = kappa;
    # homogeneous bulk free energy
    def f_0(self,phi_p,phi_r):
        return self.rho_s * (phi_p - self.c_alpha)**2 * (self.c_beta-phi_p)**2 + self.chi * phi_p * phi_r

    # free energy
    def f(self,phi_p,phi_r):
        return self.f_0(phi_p,phi_r)+ .5*self.kappa*(phi_p.grad.mag)**2

    # chemical potential
    def mu_p(self,phi_p,phi_r):
        return 2* self.rho_s * (self.c_alpha-phi_p) * (self.c_beta-phi_p) * (self.c_alpha + self.c_beta - 2*phi_p) +self.chi * phi_r

    # first derivative of chemical potential
    def dmu_p_dphi_p(self,phi_p,phi_r):
        return 2 * self.rho_s * ((self.c_alpha - phi_p)**2 + 4*(self.c_alpha - phi_p)*(self.c_beta - phi_p) + (self.c_beta - phi_p)**2)

    def dmu_p_dphi_r(self,phi_p,phi_r):
        return self.chi;

class free_energy_FH():
    """
    Simple Flory-Huggins free-energy with constant chi - provided for backwards compatibility

    Use free_energy_FH_changing_chi_LG with a=0 (same result)
    """
    def __init__(self,c_alpha,c_beta,rho_s,chi,kappa):
        self.c_alpha = c_alpha;
        self.c_beta = c_beta;
        self.rho_s = rho_s;
        self.chi_pr = chi;
        self.kappa = kappa;
        self.chi_ps = np.log(c_beta/(1-c_beta))/(2*c_beta-1);
        self.chi_rs = 0;
        print(self.chi_ps)

    # homogeneous bulk free energy
    def f_0(self,phi_p,phi_r):
        return (phi_p)*np.log(phi_p) + phi_r*np.log(phi_r) + (1-phi_p-phi_r)*np.log((1-phi_p-phi_r)) + self.chi_pr*phi_p*phi_r + self.chi_ps*phi_p*(1-phi_p-phi_r) + self.chi_rs*phi_r*(1-phi_p-phi_r)

    # free energy
    def f(self,phi_p,phi_r):
        return self.f_0(phi_p,phi_r)+ .5*self.kappa*(phi_p.grad.mag)**2

    # chemical potential
    def mu_p(self,phi_p,phi_r):
        return (np.log(phi_p/(1-phi_p-phi_r)) + self.chi_pr*phi_r + self.chi_ps*(1-2*phi_p-phi_r) - self.chi_rs*phi_r)

    def mu_r(self,phi_p,phi_r):
        return (np.log(phi_r/(1-phi_p-phi_r)) + self.chi_pr*phi_p + self.chi_rs*(1-2*phi_r-phi_p) -self.chi_ps*phi_p)

    # first derivative of chemical potential
    def dmu_p_dphi_p(self,phi_p,phi_r):
        return (1/phi_p + 1/(1-phi_p-phi_r) - 2*self.chi_ps)

    def dmu_p_dphi_r(self,phi_p,phi_r):
        return (1/(1-phi_p-phi_r) + self.chi_pr  - self.chi_ps -self.chi_rs);

    def dmu_r_dphi_r(self,phi_p,phi_r):
        return (1/phi_r + 1/(1-phi_p-phi_r)  - 2*self.chi_rs)
