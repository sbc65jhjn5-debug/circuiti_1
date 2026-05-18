import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.cost import LeastSquares
from scipy.stats import chi2

# Legge di Shockley:
q_KT = 38.6 # V^-1

def I_shockley (V, I_0, g):
    return I_0 * (np.exp (q_KT * V / g) - 1)

def dI_dV_shockley (V, I_0, g):
    return I_0 * np.exp (q_KT * V / g) * q_KT / g

# Prendiamo (come resistore) nelle due configurazioni V e I

# Configurazione 1:
#                 ----- amperometro --- resistenza ----
# generatore ----                                       -----
#                 -------------- voltmetro ------------

# Configurazione 2:
#                                  ----- resistenza -----
# generatore ---- amperometro ----                        ----
#                                  ----- voltmetro ------


# Configurazione 1: (err_I --> video 3 (dildo1.txt))

V_1 = np.array ([0.102, 0.152, 0.173, 0.203, 0.222, 0.252, 0.272, 0.302, 0.322, 0.352, 0.373, 0.403]) #V    (Da [0.352, 5.42] errore più alto su I vedi video 4)
I_1 = np.array ([0.008, 0.035, 0.058, 0.123, 0.203, 0.431, 0.705, 1.521, 2.550, 5.42, 9.16, 18.60]) #e-6 A

sigma_I_11 = np.ones (9) * 0.0205 #microA
sigma_I_12 = np.ones (3) * 0.0202 #microA
sigma_I_conf1 = np.array ([0.0205,
                           0.0205,
                           0.0205,
                           0.0205,
                           0.0205,
                           0.0205,
                           0.0205,
                           0.0205,
                           0.0205,
                           0.0202,
                           0.0202,
                           0.0202
                           ])

#Configurazione 2: (err_I --> video 5)
V_2 = np.array ([0.052, 0.072, 0.102, 0.122, 0.152, 0.173, 0.202, 0.222, 0.252, 0.272, 0.301, 0.321, 0.351, 0.372, 0.399]) # V    (Da [0.222, 0.221] errore su I video 6)
I_2 = np.array ([0.004, 0.009, 0.016, 0.025, 0.048, 0.073, 0.142, 0.221, 0.447, 0.725, 1.539, 2.555, 5.410, 9.10 , 18.50]) # e-6 A


# Fit in configurazione 1:

sigma_V_1 = 0.0005 # V
sigma_tot_conf1 = sigma_I_conf1.copy ()

for _ in range (8):
    ls = LeastSquares (V_1, 
                       I_1,
                       sigma_tot_conf1,
                       I_shockley)
    
    m = Minuit (ls, 
                I_0 = 0.0001,
                g = 1.5
                )
    
    m.migrad ()
    
    derivata_I_1 = dI_dV_shockley (V_1, m.values["I_0"], m.values["g"])
    sigma_tot_conf1 = np.sqrt (sigma_I_conf1**2 + (derivata_I_1 * sigma_V_1)**2)

ls = LeastSquares (V_1, 
                   I_1,
                   sigma_tot_conf1,
                   I_shockley)

m = Minuit (ls, 
            I_0 = m.values["I_0"],
            g = m.values["g"]
            )

m.migrad ()

for par, val, err in zip (m.parameters, m.values, m.errors):
    print (f"{par} = {val:.4f} \\pm {err:.4f}")
    
I_0_fit_1 = m.values["I_0"]
g_fit_1 = m.values["g"]

I_fit_punti_1 = I_shockley (V_1, I_0_fit_1, g_fit_1)
n_parametri_fit_1 = len (m.parameters)

chi2_lineare = np.sum (((I_1 - I_fit_punti_1) / sigma_tot_conf1) ** 2)
ndof_lineare = len (I_1) - n_parametri_fit_1
p_value_lineare = chi2.sf (chi2_lineare, ndof_lineare)

print (f"Chi2 lineare = {chi2_lineare:.2f}")
print (f"ndof lineare = {ndof_lineare}")
print (f"Chi2 ridotto lineare = {chi2_lineare / ndof_lineare:.2f}")
print (f"p-value lineare = {p_value_lineare:.4f}")

mask_log = (I_1 > 0) & (I_fit_punti_1 > 0)
sigma_log_I_1 = sigma_tot_conf1[mask_log] / I_1[mask_log]
chi2_log = np.sum (((np.log (I_1[mask_log]) - np.log (I_fit_punti_1[mask_log])) / sigma_log_I_1) ** 2)
ndof_log = np.sum (mask_log) - n_parametri_fit_1
p_value_log = chi2.sf (chi2_log, ndof_log)

print (f"Chi2 logaritmico = {chi2_log:.2f}")
print (f"ndof logaritmico = {ndof_log}")
print (f"Chi2 ridotto logaritmico = {chi2_log / ndof_log:.2f}")
print (f"p-value logaritmico = {p_value_log:.4f}")

fig, ax = plt.subplots (ncols = 2, nrows = 1, figsize = (12, 5))

x_axis_1 = np.linspace (0.07, max(V_2) + 0.01, 5000)
y_fit_1 = I_shockley (x_axis_1, I_0_fit_1, g_fit_1)
y_fit_1_log = np.where (y_fit_1 > 0, y_fit_1, np.nan)

fig.suptitle ("Caratterizzazione del diodo")
ax[0].set_title ("Scala lineare")
ax[0].set_xlabel ("Tensione (V)")
ax[0].set_ylabel ("Intensità di corrente ($\\mu$A)")

ax[0].plot (x_axis_1,
         y_fit_1,
         label = "Fit con legge di Shockley",
         color = "olivedrab"
         )

ax[0].errorbar (V_1, I_1,
             xerr = sigma_V_1,
             yerr = sigma_tot_conf1,
             color = "indigo",
             linestyle = "None",
             marker = 'o',
             capsize = 4,
             label = "Dati osservati"
             )

ax[1].set_title ("Scala logaritmica")
ax[1].set_xlabel ("Tensione (V)")
ax[1].set_ylabel ("Intensità di corrente ($\\mu$A)")
ax[1].set_yscale ("log")

ax[1].plot (x_axis_1,
         y_fit_1_log,
         label = "Fit con legge di Shockley",
         color = "olivedrab"
         )

ax[1].errorbar (V_1, I_1,
             xerr = sigma_V_1,
             yerr = sigma_tot_conf1,
             color = "indigo",
             linestyle = "None",
             marker = 'o',
             capsize = 4,
             label = "Dati osservati"
             )

for axis in ax:
    axis.legend ()
    axis.grid (True, which = "both")

plt.tight_layout ()
plt.show ()
