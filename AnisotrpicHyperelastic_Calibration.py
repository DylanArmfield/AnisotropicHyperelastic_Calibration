# -*- coding: utf-8 -*-
"""
Created on Mon May 27 12:37:45 2024

@author: dylan
"""

import numpy as np
import matplotlib.pylab as plt
import scipy.optimize as opt
import pandas as pd
from scipy import interpolate
import math
from multiprocessing.pool import ThreadPool

from pymoo.operators.sampling.lhs import LHS
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.algorithms.soo.nonconvex.nelder import NelderMead
from pymoo.algorithms.soo.nonconvex.isres import ISRES

import multiprocessing
from pymoo.core.problem import StarmapParallelization

############################ Read data #############################################

sample = []
PD_data = {}
PD_raw = {}

models = ["Muti-fibre", "GOH"]
model = models[1]
print("Chosen Model: " + model)
test_type = "Biaxial"  # "Uniaxial"
print("Test protocol: " + test_type)

sources = ["Li", "Caballero"]
source = sources[0]
 
opt_ngen = 5
# num_sample = ['03_1', '05_1', '075_1', '1_1', '1_075', '1_05', '1_03']
num_sample = ['03_1', '05_1', '075_1', '1_1', '1_075', '1_05']
# num_sample = ['075_1', '1_1', '1_075']
# num_sample = ['1_1']

################################## X1  #########################################

for i in range(len(num_sample)):
    file = './test_data/'+source+'_E11_'+ num_sample[i]+'.csv'
    
    df_pd1 = pd.read_csv(file, usecols=['Strain', 'Stress'])
    strain_grad = True
    idx = 0
    df_pd1 = df_pd1.iloc[::-1].reset_index(drop=True)
    
 
    max_str = df_pd1.idxmax(axis=0)['Stress'] 
    df_pd1 = df_pd1[df_pd1.index > max_str]  
    df_pd1 = df_pd1.iloc[::-1].reset_index(drop=True)

    
    strain_pd1 = df_pd1[["Strain"]].to_numpy()
    stress_pd1 = df_pd1[["Stress"]].to_numpy()/1000
    
    p = opt.curve_fit(lambda t,a,b,c: a+b*np.log(t+c),  stress_pd1[:,0],  strain_pd1[:,0], maxfev=100000)
    stress_ax_approx = np.logspace(np.log10(0.02), np.log10(1), 30)
    strain_ax_approx = p[0][0] + p[0][1]*np.log(stress_ax_approx+ p[0][2])
    stress_ax_approx = np.append([0],stress_ax_approx)
    strain_ax_approx = np.append([0],strain_ax_approx)
    
    PD_data.update( { ('X1_'+num_sample[i]) : { 'Strain': strain_ax_approx, 'Stress' : stress_ax_approx} } )
    PD_raw.update( { ('X1_'+num_sample[i]) : { 'Strain': strain_pd1, 'Stress' : stress_pd1} } )
    sample.append('X1_'+num_sample[i])
    plt.figure(1)
    # plt.scatter(PD_raw['X1_'+num_sample[i]]['Strain'], PD_raw['X1_'+num_sample[i]]['Stress'],label='X1_'+num_sample[i], c='blue')
    # plt.plot(PD_data['X1_'+num_sample[i]]['Strain'], PD_data['X1_'+num_sample[i]]['Stress'],label='X1_'+num_sample[i], c='red')
    

################################## X2  #########################################

for i in range(len(num_sample)):
    file = './test_data/'+source+'_E22_'+ num_sample[i]+'.csv'
    
    df_pd1 = pd.read_csv(file, usecols=['Strain', 'Stress'])
    
    strain_grad = True
    idx = 0
    df_pd1 = df_pd1.iloc[::-1].reset_index(drop=True)
    
 
    max_str = df_pd1.idxmax(axis=0)['Stress'] 
    df_pd1 = df_pd1[df_pd1.index > max_str]  
    df_pd1 = df_pd1.iloc[::-1].reset_index(drop=True)

    
    strain_pd1 = df_pd1[["Strain"]].to_numpy() 
    stress_pd1 = df_pd1[["Stress"]].to_numpy()/1000
    
    p = opt.curve_fit(lambda t,a,b,c: a+b*np.log(t+c),  stress_pd1[:,0],  strain_pd1[:,0], maxfev=100000)
    stress_ax_approx = np.logspace(np.log10(0.02), np.log10(1), 30)
    strain_ax_approx = p[0][0] + p[0][1]*np.log(stress_ax_approx+ p[0][2])
    stress_ax_approx = np.append([0],stress_ax_approx)
    strain_ax_approx = np.append([0],strain_ax_approx)
    
    PD_data.update( { ('X2_'+num_sample[i]) : { 'Strain': strain_ax_approx, 'Stress' : stress_ax_approx} } )
    PD_raw.update( { ('X2_'+num_sample[i]) : { 'Strain': strain_pd1, 'Stress' : stress_pd1} } )
    sample.append('X2_'+num_sample[i])
#     plt.figure(1)
#     plt.scatter(PD_raw['X2_'+num_sample[i]]['Strain'], PD_raw['X2_'+num_sample[i]]['Stress'],label='X2_'+num_sample[i], c='green')
#     plt.plot(PD_data['X2_'+num_sample[i]]['Strain'], PD_data['X2_'+num_sample[i]]['Stress'],label='X2_'+num_sample[i], c='black')

# plt.figure(1)
# plt.xlabel("Strain")
# plt.ylabel("Stress [MPa]")
# plt.xlim((-0.05, 0.35))


###################### Anisotrpoic Fitting Functions #####################################################

def holzapfel_direct(test_type, inputs, exp_strain11, exp_strain22, exp_strain33):
    
    if test_type == 'Uniaxial':
        stretch_1 = np.exp(exp_strain11)
        stretch_2 = stretch_1 **(-1/2)
        stretch_3 = stretch_1 **(-1/2)
    
    if test_type == 'Biaxial':
        stretch_1 = np.exp(exp_strain11)
        stretch_2 = np.exp(exp_strain22)
        stretch_3 = 1 / (stretch_1 * stretch_2)
       
    c1 = inputs[0]
    k1 = inputs[1]
    k2 = inputs[2]
    kappa = inputs[3]
    gamma1 = inputs[4]
    gamma2 = 0
    
    F = np.diag([stretch_1,stretch_2, stretch_3])
    J = np.linalg.det(F)
    F_bar = J**(-1/3) * F
    C = np.transpose(F_bar) @ F_bar
    B = F_bar @ np.transpose(F_bar)
    E = (C - np.identity(3) )/2

    D = 1e-6

    I1 = np.trace(C)
    I2 = (1/2) * (np.trace(C) ** 2 - np.trace(C ** 2))
    I3 = np.linalg.det(C)

    N1 = np.array([[np.cos(gamma1), np.sin(gamma1), 0]]).transpose()
    N2 = np.array([[np.cos(gamma2), -np.sin(gamma2), 0]]).transpose()
    
    A1 = N1 @ np.transpose(N1)
    A2 = N2 @ np.transpose(N2)
    
    I4 = np.trace(C @ A1)
    I6 = np.trace(C @ A2)
    
    W_iso = c1 * (I1 - 3) + (1/D)*(((J**2 - 1)/2) - np.log(J) )
    
    w4 = np.exp(k2 * (kappa * (I1 - 3) + (1 - 3 * kappa) * (I4 - 1) ) ** 2) - 1
    w6 = np.exp(k2 * (kappa * (I1 - 3) + (1 - 3 * kappa) * (I6 - 1) ) ** 2) - 1
    
    W_aniso = k1 / (2 * k2) * (w4) # + w6)
    
    Strain_energy = W_iso + W_aniso
   
    Sigma_iso = c1 * 2* (F_bar - (1/3)*np.trace(F_bar)*np.identity(3) ) 
    Sigma_vol = (1/D)*(J - 1/J) * (np.linalg.inv(F_bar).T * np.linalg.det(F_bar))
    
    # Uncomment the Eb term in d_psi_dI1 and the d_psi_dI6*d_I6_dF in Sigma_aniso to get a 2 fibre family HGO
    
    Ea = kappa*(I1-3) + (1-3*kappa)*(I4-1)
    Eb = kappa*(I1-3) + (1-3*kappa)*(I6-1)
    
    d_psi_dI1 = k1 / (2 * k2) * ( np.exp(k2*Ea**2)) * (2*k2*Ea) * kappa  #+ k1 / (2 * k2) * ( np.exp(k2*Eb**2)) * (2*k2*Eb) * kappa
    d_I1_dF = 2*F_bar
    
    d_psi_dI4 = k1 / (2 * k2) * ( np.exp(k2*Ea**2)) * (2*k2*Ea) * (1 -3*kappa)
    d_I4_dF = 2* F_bar @ A1
    
    d_psi_dI6 = k1 / (2 * k2) * ( np.exp(k2*Eb**2)) * (2*k2*Eb) * (1 -3*kappa)
    d_I6_dF = 2* F_bar @ A2
    
    Sigma_aniso = d_psi_dI1*d_I1_dF + d_psi_dI4*d_I4_dF #+ d_psi_dI6*d_I6_dF
    
    Sigma = Sigma_iso + Sigma_vol + Sigma_aniso  

    sigma11 = Sigma[0][0]
    sigma22 = Sigma[1][1]
    sigma33 = Sigma[2][2]
        
    return(sigma11, sigma22, sigma33)

def fourfibre_direct(test_type, inputs, exp_strain11, exp_strain22, exp_strain33):
    
    if test_type == 'Uniaxial':
        stretch_1 = np.exp(exp_strain11)
        stretch_2 = stretch_1 **(-1/2)
        stretch_3 = stretch_1 **(-1/2)
    
    if test_type == 'Biaxial':
        stretch_1 = np.exp(exp_strain11)
        stretch_2 = np.exp(exp_strain22)
        stretch_3 = 1 / (stretch_1 * stretch_2)
       
    c1 = inputs[0]
    k1 = inputs[1]
    k2 = inputs[2]
    k3 = inputs[3]
    k4 = inputs[4]
    gamma1 = inputs[5]
    gamma2 = gamma1 + np.radians(90)
    
    F = np.diag([stretch_1,stretch_2, stretch_3])
    J = np.linalg.det(F)
    F_bar = J**(-1/3) * F
    C = np.transpose(F_bar) @ F_bar
    B = F_bar @ np.transpose(F_bar)
    E = (C - np.identity(3) )/2

    D = 1e-8

    I1 = np.trace(C)
    I2 = (1/2) * (np.trace(C) ** 2 - np.trace(C ** 2))
    I3 = np.linalg.det(C)

    N1 = np.array([[np.cos(0), np.sin(0), 0]]).transpose()
    # N1 = np.array([[np.cos(gamma1), np.sin(gamma1), 0]]).transpose()
    N2 = np.array([[np.cos(0), np.sin(0), 0]]).transpose()
    
    Nl = np.array([[np.cos(gamma1), np.sin(gamma1), 0]]).transpose()
    Nz = np.array([[np.cos(gamma2), np.sin(gamma2), 0]]).transpose()
    # Nl = np.array([[1, 0, 0]]).transpose()
    # Nz = np.array([[0, 1, 0]]).transpose()
    
    A1 = N1 @ np.transpose(N1)
    A2 = N2 @ np.transpose(N2)
    Al = Nl @ np.transpose(Nl)
    Az = Nz @ np.transpose(Nz)
    
    I4 = np.trace(C @ A1)
    I6 = np.trace(C @ A2)
    Il = np.trace(C @ Al)
    Iz = np.trace(C @ Az)
    
    W_iso = c1 * (I1 - 3) + (1/D)*(((J**2 - 1)/2) - np.log(J) )
    
    w4 = np.exp(k2 * (I4 - 1) ** 2) - 1
    w6 = np.exp(k2 * (I6 - 1) ** 2) - 1
    
    wl = np.exp(k4 * (Il - 1) ** 2) - 1
    wz = np.exp(k4 * (Iz - 1) ** 2) - 1
    
    W_aniso = (k1 / (4 * k2)) * (w4) + (k3 / (4 * k4)) * ( wl + wz )
    
    Strain_energy = W_iso + W_aniso
   
    Sigma_iso = c1 * 2* (F_bar - (1/3)*np.trace(F_bar)*np.identity(3) ) 
    Sigma_vol = (1/D)*(J - 1/J) * (np.linalg.inv(F_bar).T * np.linalg.det(F_bar))
    
    
    d_psi_dI4 = (k1/2) * np.exp(k2 * (I4 - 1)**2 ) * (I4 - 1)
    d_I4_dF = 2* F_bar @ A1
    
    d_psi_dI6 = (k1/2) * np.exp(k2 * (I6 - 1)**2 ) * (I6 - 1)
    d_I6_dF = 2* F_bar @ A2
    
    d_psi_dIl = (k3/2) * np.exp(k4 * (Il - 1)**2 ) * (Il - 1)
    d_Il_dF = 2* F_bar @ Al
    
    d_psi_dIz = (k3/2) * np.exp(k4 * (Iz - 1)**2 ) * (Iz - 1)
    d_Iz_dF = 2* F_bar @ Az
    
    Sigma_aniso = d_psi_dI4*d_I4_dF +  d_psi_dIl*d_Il_dF + d_psi_dIz*d_Iz_dF  # #d_psi_dI6*d_I6_dF 
    
    Sigma = Sigma_iso + Sigma_vol + Sigma_aniso  

    sigma11 = Sigma[0][0]
    sigma22 = Sigma[1][1]
    sigma33 = Sigma[2][2]
        
    return(sigma11, sigma22, sigma33)

####################################################### Problem definitions ######################################################################################################

class GOH_Problem(ElementwiseProblem):

    def __init__(self, elementwise_runner=None, **kwargs):
        super().__init__(n_var=5,
                         n_obj=1,
                         xl=[0.1, 1E-6, 1E-6, 0, np.radians(45)],
                         xu=[5, 1000, 1000, 1/3, np.radians(45)])
        self.elementwise_runner = elementwise_runner

    def _evaluate(self, x, out, *args, **kwargs):

        error = 0
        
        for s in range(int(len(sample)/2)):
            
            s1 = s + int(len(sample)/2)
            
            stress_dir1 = np.zeros((len(PD_data[str(sample[s])]['Strain'])) )
            stress_dir2 = np.zeros((len(PD_data[str(sample[s])]['Strain'])) )
            stress_dir3 = np.zeros((len(PD_data[str(sample[s])]['Strain'])) )

            # strain22 = np.zeros(len(PD_data[str(sample[s])]['Strain']) )
            strain33 = np.zeros(len(PD_data[str(sample[s])]['Strain']) )
            
            for i in range(len(PD_data[str(sample[s])]['Strain']) ):     
                stress_dir1[i], stress_dir2[i], stress_dir3[i] = holzapfel_direct(test_type, x, PD_data[str(sample[s])]['Strain'][i], PD_data[str(sample[s1])]['Strain'][i], strain33[i])
        
        
            error += np.sqrt((1/2*len(stress_dir1)) * np.sum( ( stress_dir1 - PD_data[str(sample[s])]['Stress'] )**2) ) #+
                        # np.sum( (strain_stress_circum[1,:] - stress_dir2)**2 /len(stress_dir2) ) )
        
        # error = np.sqrt(np.sum( (PD_data['PD'+str(s)]['Stress'] - stress_dir1)**2 /len(stress_dir1) ) )
        
        out["F"] = error

class fourfibre_Problem(ElementwiseProblem):

    def __init__(self, elementwise_runner=None, **kwargs):
        super().__init__(n_var=6,
                         n_obj=1,
                         xl=[0.1,        1,        1,    0.01,     1,   np.radians(0)],
                         xu=[  5,      100,      300,     100,   200,   np.radians(90)])
        self.elementwise_runner = elementwise_runner

    def _evaluate(self, x, out, *args, **kwargs):

        error = 0
        
        for s in range(int(len(sample)/2)):
            
            s1 = s + int(len(sample)/2)
            
            stress_dir1 = np.zeros((len(PD_data[str(sample[s])]['Strain'])) )
            stress_dir2 = np.zeros((len(PD_data[str(sample[s])]['Strain'])) )
            stress_dir3 = np.zeros((len(PD_data[str(sample[s])]['Strain'])) )

            # strain22 = np.zeros(len(PD_data[str(sample[s])]['Strain']) )
            strain33 = np.zeros(len(PD_data[str(sample[s])]['Strain']) )
            
            for i in range(len(PD_data[str(sample[s])]['Strain']) ):     
                stress_dir1[i], stress_dir2[i], stress_dir3[i] = fourfibre_direct(test_type, x, PD_data[str(sample[s])]['Strain'][i], PD_data[str(sample[s1])]['Strain'][i], strain33[i])
        
        
            error += np.sqrt((1/2*len(stress_dir1)) * np.sum( ( stress_dir1 - PD_data[str(sample[s])]['Stress'] )**2) ) #+
                        # np.sum( (strain_stress_circum[1,:] - stress_dir2)**2 /len(stress_dir2) ) )
        
        # error = np.sqrt(np.sum( (PD_data['PD'+str(s)]['Stress'] - stress_dir1)**2 /len(stress_dir1) ) )
        
        out["F"] = error


##########################################################################################################################################################################################
# Serial execcution:
# algorithm = GA(pop_size=60,eliminate_duplicates=True)
# problem = MyProblem()
# res = minimize(problem,algorithm,seed=1,verbose=True)

### Parallel execcution:
### initialize the thread pool and create the runner
n_threads = 100
pool = ThreadPool(n_threads)
runner = StarmapParallelization(pool.starmap)

### define the problem by passing the starmap interface of the thread pool
if model == "Muti-fibre":
    problem = fourfibre_Problem(elementwise_runner=runner)
    
    res = minimize(problem, GA(pop_size=60,eliminate_duplicates=True), termination=("n_gen", opt_ngen), seed=1, verbose=True)
    print('Threads:', res.exec_time)
    pool.close()

    print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))
    print("Fibre directions: ")
    print(np.array([np.cos(0), np.sin(0), 0]))
    print(np.array([[np.cos(res.X[5]), np.sin(res.X[5]), 0]]) )
    print(np.array([[np.cos(res.X[5] + np.radians(90)), np.sin(res.X[5] + np.radians(90)), 0]]) )
    
    
elif model == "GOH":
    problem = GOH_Problem(elementwise_runner=runner)
    
    res = minimize(problem, GA(pop_size=60,eliminate_duplicates=True), termination=("n_gen", opt_ngen), seed=1, verbose=True)
    print('Threads:', res.exec_time)
    pool.close()

    print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))
    print("Fibre directions: ")
    print(np.array([[np.cos(res.X[4]), np.sin(res.X[4]), 0]]) )

# inputs_fin = [1, 1.2928, 3.5417, 0.2584, 15.026, 0.997]
# inputs_fin = [ 0.42044937,  0.09580622, 30.60876605,  0.35308313 , 3.71450354 , 0.78529759]
inputs_fin = res.X

for s in range(int(len(sample)/2)):
    # gamma = np.radians(0) 
    s1 = s + int(len(sample)/2)
          
    
    s11 = np.zeros((len(PD_data[str(sample[s])]['Strain'])) )
    s22 = np.zeros((len(PD_data[str(sample[s])]['Strain'])) )
    s33 = np.zeros((len(PD_data[str(sample[s])]['Strain'])) )
    
    strain33 = np.zeros(len(PD_data[str(sample[s])]['Strain']) )
    
    if model == "Muti-fibre":
        for i in range(len(PD_data[str(sample[s])]['Strain'])):     
            s11[i], s22[i], s33[i] = fourfibre_direct(test_type, inputs_fin, PD_data[str(sample[s])]['Strain'][i], PD_data[str(sample[s1])]['Strain'][i], strain33[i])
    
    elif model == "GOH":
        for i in range(len(PD_data[str(sample[s])]['Strain'])):     
            s11[i], s22[i], s33[i] = holzapfel_direct(test_type, inputs_fin, PD_data[str(sample[s])]['Strain'][i], PD_data[str(sample[s1])]['Strain'][i], strain33[i])
        
    plt.figure(2)
    plt.scatter((PD_raw[str(sample[s])]['Strain']), (PD_raw[str(sample[s])]['Stress']),marker='o', color='black', facecolors='none')
    plt.plot((PD_data[str(sample[s])]['Strain']), s11, color='blue')
    
    plt.figure(3)
    plt.scatter((PD_raw[str(sample[s1])]['Strain']), (PD_raw[str(sample[s1])]['Stress']), marker='o', color='black', facecolors='none')
    plt.plot((PD_data[str(sample[s1])]['Strain']), s11, color='blue')     
    
    # plt.scatter((PD_data[str(sample[s])]['Strain']), (PD_data[str(sample[s])]['Stress']), color='red')
    # plt.scatter((PD_data[str(sample[s1])]['Strain']), (PD_data[str(sample[s1])]['Stress']), color='blue')
    



# ##################################################### Plotting ############################################

plt.figure(2)
plt.xlabel("$E$")
plt.ylabel("$S_{11}$ [MPa]")
plt.ylim((0, 1.5))
plt.xlim((-0.05, 0.20))
plt.legend(['Exp data', model +' $E_{11}$ fit'])
# plt.savefig(model+'_'+source+'_multifit_E11.png', dpi=300)

plt.figure(3)
plt.xlabel("$E_{22}$")
plt.ylabel("$S_{11}$ [MPa]")
plt.ylim((0, 1.5))
plt.xlim((-0.05, 0.35))
plt.legend(['Exp data', model + ' $E_{22}$ fit'])
# plt.savefig(model+'_'+source+'_multifit_E22.png', dpi=300)