import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.cost import LeastSquares
from scipy.stats import chi2

# Legge di Shockley:
q_KT = 38.6 # V^-1

def I_shockley (V, I_0, g, c):
    return I_0 * (np.exp (q_KT * V / g) - 1) + c

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

sigma_I_11 = np.ones (9) * 0.00205 #microA
sigma_I_12 = np.ones (3) * 0.00202 #microA
sigma_I_conf1 = np.array ([0.00205,
                           0.00205,
                           0.00205,
                           0.00205,
                           0.00205,
                           0.00205,
                           0.00205,
                           0.00205,
                           0.00205,
                           0.00202,
                           0.00202,
                           0.00202
                           ])

#Configurazione 2: (err_I --> video 5)
V_2 = np.array ([0.052, 0.072, 0.102, 0.122, 0.152, 0.173, 0.202, 0.222, 0.252, 0.272, 0.301, 0.321, 0.351, 0.372, 0.399]) # V    (Da [0.222, 0.221] errore su I video 6)
I_2 = np.array ([0.004, 0.009, 0.016, 0.025, 0.048, 0.073, 0.142, 0.221, 0.447, 0.725, 1.539, 2.555, 5.410, 9.10 , 18.50]) # e-6 A


# Fit in configurazione 1:

ls = LeastSquares (V_1, 
                   I_1,
                   sigma_I_conf1,
                   I_shockley)

m = Minuit (ls, 
            I_0 = 0.0001,
            g = 1.5,
            c = 0
            )

m.migrad ()

for par, val, err in zip (m.parameters, m.values, m.errors):
    print (f"{par} = {val:.4f} \pm {err:.4f}")
    
I_0_fit_1 = m.values["I_0"]
g_fit_1 = m.values["g"]
c_fit_1 = m.values["c"]

fig, ax = plt.subplots ()

x_axis_1 = np.linspace (0.07, max(V_2) + 0.01, 5000)

ax.set_title ("Caratterizzazione del diodo")
ax.set_xlabel ("Tensione (V)")
ax.set_ylabel ("Intensità di corrente ($\\mu$A)")

ax.plot (x_axis_1,
         [I_shockley(x, I_0_fit_1, g_fit_1, c_fit_1) for x in x_axis_1],
         label = "Fit con legge di Shockley",
         color = "olivedrab"
         )

ax.errorbar (V_1, I_1,
             yerr = sigma_I_conf1,
             color = "indigo",
             linestyle = "None",
             marker = 'o',
             capsize = 4,
             label = "Dati osservati"
             )

plt.legend ()
plt.grid (True)
plt.show ()