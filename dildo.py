import numpy as np

# Prendiamo (come resistore) nelle due configurazioni V e I

# Configurazione 1:
#                 ----- amperometro --- resistenza ----
# generatore ----                                       -----
#                 -------------- voltmetro ------------

# Configurazione 2:
#                                  ----- resistenza -----
# generatore ---- amperometro ----                        ----
#                                  ----- voltmetro ------


# Configurazione 1: (err_I --> video 3)

V_1 = np.array ([0.102, 0.152, 0.173, 0.203, 0.222, 0.252, 0.272, 0.302, 0.322, 0.352, 0.373, 0.403]) #V    (Da [0.352, 5.42] errore più alto su I vedi video 4)
I_1 = np.array ([0.008, 0.035, 0.058, 0.123, 0.203, 0.431, 0.705, 1.521, 2.550, 5.42, 9.16, 18.60]) #e-6 A


#Configurazione 2: (err_I --> video 5)
V_2 = np.array ([0.052, 0.072, 0.102, 0.122, 0.152, 0.173, 0.202, 0.222, 0.252, 0.272, 0.301, 0.321, 0.351, 0.372, 0.399]) # V    (Da [0.222, 0.221] errore su I video 6)
I_2 = np.array ([0.004, 0.009, 0.016, 0.025, 0.048, 0.073, 0.142, 0.221, 0.447, 0.725, 1.539, 2.555, 5.410, 9.10 , 18.50]) # e-6 A