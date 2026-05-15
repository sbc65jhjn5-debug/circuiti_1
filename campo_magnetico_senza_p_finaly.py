import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.cost import LeastSquares
from scipy.stats import chi2

# funzioni per fit 'Bobina':

def B_bobina (N, L, I, r, offset) :
    return (4 * np.pi * 1e-7) * N * I / (2* np.sqrt (r**2 + (L/2)**2)) + offset

def tan_angoli (I, N, L, r, offset, B_t):
    B_b = B_bobina (N, L, I, r, offset)
    return B_b / B_t

def sigma_theta (sigma_corr, I, B_fit, r, sigma_r, L, sigma_L, N):
    dtheta_dI =  1/ B_fit * (4 * np.pi * 1e-7) * N / (2* np.sqrt (r**2 + (L/2)**2))
    dtheta_dr = 1 / B_fit * (8 * np.pi * 1e-7) * N * I * r / (4 * np.sqrt(r**2 + (L/2)**2)* (r**2 + (L/2)**2))
    dtheta_dL = 1 / B_fit * (2 * np.pi * 1e-7) * N * I * L / (4 * np.sqrt(r**2 + (L/2)**2)* (r**2 + (L/2)**2))
    sigma = np.sqrt ((dtheta_dI * sigma_corr)**2 + (dtheta_dr * sigma_r)**2 + (dtheta_dL * sigma_L)**2)
    return sigma

'''
# fit 'Spire':

def B_spire (N, r, I):
    return (4 * np.pi * 1e-7) * N * I / (2 * np.pi * r)

def tan_angoli_spire (I, N, r, B_t):
    B_s = B_spire (N, r, I)
    return B_s / B_t

def sigma_theta_I_spire (sigma_corr, theta, B_fit, r, N):
    dI = 2 * np.pi * r * B_fit / (4 * np.pi * 1e-7 * N ) * 1 / (1 + theta**2)
    sigma = dI * sigma_corr
    return sigma

VIENE MEGLIO CON L'ALTRO OLE
'''


deg_spento = np.array ([85, 86, 87, 90, 90, 90, 91, 92, 86, 91, 92, 93, 91, 92, 93, 90])
print (np.mean (deg_spento))

lunghezza = 5.06e-2 # m
raggio = 25.00e-2 / 2 # m

# per errore di I vedere video Ampere
I = np.array ([0.10200, 
               0.15025, 
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
               0.75565
               ])

sigma_corrente = 0.00003 # DA INSERIRE IL VALORE CORRETTO

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


deg = np.array (np.mean (deg_spento) * np.ones (14) - [np.mean(deg_1), 
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
                 np.mean(deg_14)
                 ])

for angolo in deg: print (f"{angolo:.2f}°")

# sigmi:
sigma_deg = np.array (np.std (deg_spento)/np.sqrt(len(deg_spento)) * np.ones (14) + [np.std(deg_1, ddof = 1)/np.sqrt(len(deg_1)), 
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
                       np.std(deg_14, ddof = 1)/np.sqrt(len(deg_14))
                      ])

sigma_deg = sigma_deg + (np.ones(14) * (1.0/np.sqrt(12)))

for sigma in sigma_deg: print (f"{sigma:.2f}°")

tangenti = np.tan (np.radians (deg))
sigma_tangenti = np.radians(sigma_deg) * (1 / np.cos(np.radians(deg)))**2

# Con formula per bobina:

ls1 = LeastSquares (I, tangenti, sigma_tangenti, tan_angoli)

m1 = Minuit (ls1, N = 31, L = lunghezza, r = raggio, offset = 0, B_t = 2e-5)
m1.fixed["N"] = True
m1.limits["L"] = (lunghezza - 0.001, lunghezza + 0.001)
m1.limits["r"] = (raggio - 0.001, raggio + 0.001)

m1.migrad ()

for par, val, err in zip (m1.parameters, m1.values, m1.errors) :
    print (f"{par} = {val:.5e} ± {err:.5e}")

B_fit = m1.values["B_t"]

sigma_tot = np.sqrt (np.array ([sigma_theta (sigma_corrente, I_val, B_fit, raggio, 0.001, lunghezza, 0.001, N = 31)**2 for I_val in I]) + sigma_tangenti**2)

ls2 = LeastSquares (I, tangenti, sigma_tot, tan_angoli)

m2 = Minuit (ls2, N = 31, L = lunghezza, r = raggio, offset = 0, B_t = 2e-5)
m2.fixed["N"] = True
m2.limits["L"] = (lunghezza - 0.001, lunghezza + 0.001)
m2.limits["r"] = (raggio - 0.001, raggio + 0.001)

m2.migrad ()

for par, val, err in zip (m2.parameters, m2.values, m2.errors) :
    print (f"{par} = {val:.5e} ± {err:.5e}")

chi_2 = m2.fval
ndof = m2.ndof
p_value = chi2.sf (chi_2, ndof)

B_fit_2 = m2.values["B_t"]
B_fit_errore = m2.errors["B_t"]

offset_fit = m2.values["offset"]

print (f"Otteniamo: B terrestre = {B_fit_2:.5e} ± {B_fit_errore:.5e}")
print (f"chi 2: {chi_2}\nndof: {ndof}\np value:{p_value}")

# grafico

fig, ax = plt.subplots ()

ax.set_title ("Fit per stimare il campo magnetico terrestre")
ax.set_xlabel ("intensità di corrente $I$ (A)")
ax.set_ylabel ("tan $\\theta$")

ax.errorbar (I, tangenti,
             yerr = sigma_tot,
             xerr = 0.00003, # DA INSERIRE IL VALORE CORRETTO,
             marker = "o",
             linestyle = "None",
             capsize = 4,
             color = "indigo",
             label = "Dati osservati"
             )

ax.plot (I, tan_angoli (I, 31, lunghezza, raggio, offset_fit, B_fit_2),
         color = "crimson",
         label = "$\\tan \\theta = \\frac{B_{bobina}}{B_{terrestre}}$ "
         )

plt.legend (fontsize=13)
plt.grid (True)
plt.show ()


# Grafico dei residui 

residui = tangenti - tan_angoli (I, 31, lunghezza, raggio, offset_fit, B_fit_2)

fig, ax = plt.subplots ()
ax.set_title ("Residui del fit per il campo magnetico terrestre")
ax.errorbar (I, residui,
             yerr = sigma_tot,
             xerr = 0.00003,
             marker = "^",
             linestyle = "None",
             capsize = 4,
             color = "indigo"
             )

plt.axhline (0, color = "crimson", linestyle = "--")
plt.grid (True)
plt.show ()
'''
# formula per SPIRE:

ls3 = LeastSquares (I, tangenti, sigma_tangenti, tan_angoli_spire)

m3 = Minuit (ls3, N = 31, r = raggio, B_t = 2e-5)
m3.fixed["N"] = True
m3.fixed["r"] = True

m3.migrad ()

for par, val, err in zip (m3.parameters, m3.values, m3.errors) :
    print (f"{par} = {val:.5e} ± {err:.5e}")

B_fit_3 = m3.values["B_t"]

sigma_tot_spire = np.sqrt (np.array ([sigma_theta_I_spire (sigma_corrente, np.radians (th), B_fit, raggio, N = 31)**2 for th in deg]) + sigma_tangenti**2)

ls4 = LeastSquares (I, tangenti, sigma_tot_spire, tan_angoli_spire)

m4 = Minuit (ls4, N = 31, r = raggio, B_t = B_fit_3)
m4.fixed["N"] = True
m4.fixed["r"] = True

m4.migrad ()

for par, val, err in zip (m4.parameters, m4.values, m4.errors) :
    print (f"{par} = {val:.5e} ± {err:.5e}")

chi_2_4 = m4.fval
ndof_4 = m4.ndof
p_value_4 = chi2.sf (chi_2, ndof)

B_fit_4 = m4.values["B_t"]
B_fit_4_errore = m4.errors["B_t"]

print (f"Otteniamo: B terrestre = {B_fit_4:.5e} ± {B_fit_4_errore:.5e}")
print (f"chi 2: {chi_2_4}\nndof: {ndof_4}\np value:{p_value_4}")

# grafico

fig, ax = plt.subplots ()

ax.set_title ("EH boh...")
ax.set_xlabel ("intensità di corrente $I$ (A)")
ax.set_ylabel ("tan $\\theta$")

ax.errorbar (I, tangenti,
             yerr = sigma_tangenti,
             xerr = 0.00003, # DA INSERIRE IL VALORE CORRETTO,
             marker = "o",
             linestyle = "None",
             capsize = 4,
             color = "indigo",
             label = "Dati osservati"
             )

ax.plot (I, tan_angoli_spire (I, 31, raggio, B_fit_4),
         color = "royalblue")

plt.show ()
'''