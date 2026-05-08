import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.cost import LeastSquares
from scipy.stats import chi2

def B_bobina (N, L, I, r) :
    return (4 * np.pi * 1e-7) * N * I / 2* np.sqrt (r**2 + (L/2)**2)

def tan_angoli (I, N, L, r, B_t):
    B_b = B_bobina (N, L, I, r)
    return B_b / B_t

def sigma_I (sigma_theta, theta, B_fit, r, L, N):
    dI = 2 * B_fit * np.sqrt (r**2 + L**2/4) / (4 * np.pi * 1e-7 * N) * 1 / (1 + theta**2)
    sigma = dI * sigma_theta
    return sigma

deg_spento = np.array ([85, 86, 87, 90, 90, 90, 91, 92, 86, 91, 92, 93, 91, 92, 93, 90])
print (np.mean (deg_spento))

lunghezza = 5.06e-2 # m
raggio = 25.00e-2 / 2 # m

# per errore di I vedere video Ampere
I = np.array ([0.10200, 
               0.10525, 
               0.20275, 
               0.25310, 
               0.30336, 
               0.35346, 
               0.40473, 
               0.45494, 
               0.50523, 
               0.55570, 
               0.60600,
               0.65598,
               0.70580,
               0.75565,
               0.80595,
               0.85590,
               0.90600,
               0.95585,
               1.00593
               ])


deg_1 = np.array([61, 61, 61, 60, 61, 61, 61, 61, 62, 62, 63, 63, 64, 64, 63, 64, 64, 63, 64, 63, 62])
deg_2 = np.array([52, 53, 54])
deg_3 = np.array([45, 46, 47])
deg_4 = np.array([40, 40, 41])
deg_5 = np.array([36, 37])
deg_6 = np.array([32, 33])
deg_7 = np.array([29, 29, 30])
deg_8 = np.array([27, 27, 28])
deg_9 = np.array([25, 26])
deg_10 = np.array([24, 24, 25])
deg_11 = np.array([23, 23, 23, 24])
deg_12 = np.array([22, 22, 22, 21])
deg_13 = np.array([20, 20, 21])
deg_14 = np.array([20, 20, 19])
deg_15 = np.array([19, 19, 19, 18])
deg_16 = np.array([18]) # sigma 0
deg_17 = np.array([18, 18, 18, 19])
deg_18 = np.array([17]) # sigma 0
deg_19 = np.array([16, 16, 16, 17]) 

deg = np.array (np.mean (deg_spento) * np.ones (19) - [np.mean(deg_1), 
                 np.mean(deg_2), 
                 np.mean(deg_3), 
                 np.mean(deg_4),
                 np.mean(deg_5), 
                 np.mean(deg_6), 
                 np.mean(deg_7), 
                 np.mean(deg_8),
                 np.mean(deg_9), 
                 np.mean(deg_10),
                 np.mean(deg_11),
                 np.mean(deg_12),
                 np.mean(deg_13),
                 np.mean(deg_14),
                 np.mean(deg_15),
                 np.mean(deg_16),  # sigma 0 
                 np.mean(deg_17),
                 np.mean(deg_18), # sigma 0
                 np.mean(deg_19)
                 ])

for angolo in deg: print (f"{angolo:.2f}°")

#sigmi:
sigma_deg = np.array (np.std (deg_spento)/np.sqrt(len(deg_spento)) * np.ones (19) + [np.std(deg_1, ddof = 1)/np.sqrt(len(deg_1)), 
                       np.std(deg_2, ddof = 1)/np.sqrt(len(deg_2)), 
                       np.std(deg_3, ddof = 1)/np.sqrt(len(deg_3)), 
                       np.std(deg_4, ddof = 1)/np.sqrt(len(deg_4)),
                       np.std(deg_5, ddof = 1)/np.sqrt(len(deg_5)), 
                       np.std(deg_6, ddof = 1)/np.sqrt(len(deg_6)), 
                       np.std(deg_7, ddof = 1)/np.sqrt(len(deg_7)), 
                       np.std(deg_8, ddof = 1)/np.sqrt(len(deg_8)),
                       np.std(deg_9, ddof = 1)/np.sqrt(len(deg_9)), 
                       np.std(deg_10, ddof = 1)/np.sqrt(len(deg_10)),
                       np.std(deg_11, ddof = 1)/np.sqrt(len(deg_11)),
                       np.std(deg_12, ddof = 1)/np.sqrt(len(deg_12)),
                       np.std(deg_13, ddof = 1)/np.sqrt(len(deg_13)),
                       np.std(deg_14, ddof = 1)/np.sqrt(len(deg_14)),
                       np.std(deg_15, ddof = 1)/np.sqrt(len(deg_15)),
                       0,
                       np.std(deg_17, ddof = 1)/np.sqrt(len(deg_17)),
                       0,
                       np.std(deg_19, ddof = 1)/np.sqrt(len(deg_19))
                       ])

for sigma in sigma_deg: print (f"{sigma:.2f}°")

tangenti = np.tan (np.radians (deg))
sigma_tangenti = sigma_deg * (1 / np.cos (np.radians (deg)))**2

ls1 = LeastSquares (I, tangenti, sigma_tangenti, tan_angoli)

m1 = Minuit (ls1, N = 31, L = lunghezza, r = raggio, B_t = 2e-5)
m1.fixed["N"] = True
m1.fixed["L"] = True
m1.fixed["r"] = True

m1.migrad ()

for par, val, err in zip (m1.parameters, m1.values, m1.errors) :
    print (f"{par} = {val:.5e} ± {err:.5e}")

B_fit = m1.values["B_t"]

sigma_tot = np.sqrt (np.array ([sigma_I (sigma_d, th, B_fit, raggio, lunghezza, N = 31)**2 for sigma_d, th in zip (sigma_deg, deg)]) + sigma_tangenti**2)

ls2 = LeastSquares (I, tangenti, sigma_tot, tan_angoli)

m2 = Minuit (ls2, N = 31, L = lunghezza, r = raggio, B_t = 2e-5)
m2.fixed["N"] = True
m2.fixed["L"] = True
m2.fixed["r"] = True

m2.migrad ()

for par, val, err in zip (m2.parameters, m2.values, m2.errors) :
    print (f"{par} = {val:.5e} ± {err:.5e}")

chi_2 = m2.fval
ndof = m2.ndof
p_value = chi2.sf (chi_2, ndof)

B_fit_2 = m2.values["B_t"]
B_fit_errore = m2.errors["B_t"]

print (f"Otteniamo: B terrestre = {B_fit_2:.5e} ± {B_fit_errore:.5e}")
print (f"chi 2: {chi_2}\nndof: {ndof}\np value:{p_value}")