import numpy as np
import matplotlib.pyplot as plt

with open ("errori_milli.txt", "r") as f:
    errori_milli = np.array ([float (x) for x in f.read ().split ()])

I_medio = np.mean (errori_milli)
I_std = np.std (errori_milli, ddof = 1)
print (f"Valore medio di I: {I_medio:.5f} mA")
print (f"Deviazione standard di I (errore da utilizzare per misure in mA): {I_std:.5f} mA")

n_bins_1 = 15
bin_edges_1 = np.linspace (min (errori_milli), max (errori_milli), n_bins_1 + 1)

fig, ax = plt.subplots ()

ax.set_title ("Distribuzione del valore di $I$ per una misura generica")
ax.hist (errori_milli, bins = bin_edges_1, color = "lightblue")
ax.set_xlabel ("intensità di corrente (mA)")
ax.set_ylabel ("frequenza")

plt.show()

# errori microoo

with open ("errori_micro.txt", "r") as f:
    errori_micro = np.array ([float (x) for x in f.read ().split ()])

I_medio = np.mean (errori_micro)
I_std = np.std (errori_micro, ddof = 1)
print (f"Valore medio di I: {I_medio:.5f} $\\mu$A")
print (f"Deviazione standard di I (errore da utilizzare per misure in $\\mu$A): {I_std:.5f} $\\mu$A")

n_bins_1 = 12
bin_edges_1 = np.linspace (min (errori_micro), max (errori_micro), n_bins_1 + 1)

fig, ax = plt.subplots ()

ax.set_title ("Distribuzione del valore di $I$ per una misura generica")
ax.hist (errori_micro, bins = bin_edges_1, color = "bisque")
ax.set_xlabel ("intensità di corrente ($\\mu$A)")
ax.set_ylabel ("frequenza")

plt.show()

# errori dildo 1 (configurazione 1)

with open ("errori_dildo1.txt", "r") as f:
    errori_dildo1 = np.array ([float (x) for x in f.read ().split ()])

I_medio = np.mean (errori_dildo1)
I_std = np.std (errori_dildo1, ddof = 1)
print (f"Valore medio di I: {I_medio:.5f} $\\mu$A")
print (f"Deviazione standard di I (errore da utilizzare per misure in $\\mu$A dildo 1): {I_std:.5f} $\\mu$A")

n_bins_1 = 8
bin_edges_1 = np.linspace (min (errori_dildo1), max (errori_dildo1), n_bins_1 + 1)

fig, ax = plt.subplots ()

ax.set_title ("Distribuzione del valore di $I$ per una misura generica")
ax.hist (errori_dildo1, bins = bin_edges_1, color = "pink")
ax.set_xlabel ("intensità di corrente ($\\mu$A)")
ax.set_ylabel ("frequenza")

plt.show()

# errori dildo 2 (punti finali della configurazione 1)

with open ("errori_dildo_2.txt", "r") as f:
    errori_dildo2 = np.array ([float (x) for x in f.read ().split ()])

I_medio = np.mean (errori_dildo2)
I_std = np.std (errori_dildo2, ddof = 1)
print (f"Valore medio di I: {I_medio:.5f} $\\mu$A")
print (f"Deviazione standard di I (errore da utilizzare per misure in $\\mu$A dildo 2): {I_std:.5f} $\\mu$A")

n_bins_1 = 7
bin_edges_1 = np.linspace (min (errori_dildo2), max (errori_dildo2), n_bins_1 + 1)

fig, ax = plt.subplots ()

ax.set_title ("Distribuzione del valore di $I$ per una misura generica")
ax.hist (errori_dildo2, bins = bin_edges_1, color = "lemonchiffon")
ax.set_xlabel ("intensità di corrente ($\\mu$A)")
ax.set_ylabel ("frequenza")

plt.show()

# errori dildo 3 (configurazione 2, punti iniziali)

with open ("errori_dildo_3.txt", "r") as f:
    errori_dildo3 = np.array ([float (x) for x in f.read ().split ()])

I_medio = np.mean (errori_dildo3)
I_std = np.std (errori_dildo3, ddof = 1)
print (f"Valore medio di I: {I_medio:.5f} $\\mu$A")
print (f"Deviazione standard di I (errore da utilizzare per misure con $\\mu$A dildo 3): {I_std:.5f} $\\mu$A")

n_bins = 9
bin_edges = np.linspace (min (errori_dildo3), max (errori_dildo3), n_bins + 1)

fig, ax = plt.subplots ()

ax.set_title ("Distribuzione del valore di $I$ per una misura generica")
ax.hist (errori_dildo3, bins = bin_edges, color = "lightsteelblue")
ax.set_xlabel ("intensità di corrente ($\\mu$A)")
ax.set_ylabel ("frequenza")

plt.show()


# errori dildo 4 (configurazione 2, punti finali)

with open ("errori_dildo_4.txt", "r") as f:
    errori_dildo4 = np.array ([float (x) for x in f.read ().split ()])

I_medio = np.mean (errori_dildo4)
I_std = np.std (errori_dildo4, ddof = 1)
print (f"Valore medio di I: {I_medio:.5f} $\\mu$A")
print (f"Deviazione standard di I (errore da utilizzare per misure con $\\mu$A dildo 4): {I_std:.5f} $\\mu$A")

n_bins = 8
bin_edges = np.linspace (min (errori_dildo4), max (errori_dildo4), n_bins + 1)

fig, ax = plt.subplots ()

ax.set_title ("Distribuzione del valore di $I$ per una misura generica")
ax.hist (errori_dildo4, bins = bin_edges, color = "paleturquoise")
ax.set_xlabel ("intensità di corrente ($\\mu$A)")
ax.set_ylabel ("frequenza")

plt.show()


# errori corrente bobina

with open ("errori_corrente_bobina.txt", "r") as f:
    errori_bobina = np.array ([float (x) for x in f.read ().split ()])

I_medio = np.mean (errori_bobina)
I_std = np.std (errori_bobina, ddof = 1)
print (f"Valore medio di I: {I_medio:.5f} A")
print (f"Deviazione standard di I (errore da utilizzare per misure della bobina in A): {I_std:.5f} A")

n_bins_1 = 9
bin_edges_1 = np.linspace (min (errori_bobina), max (errori_bobina), n_bins_1 + 1)

fig, ax = plt.subplots ()

ax.set_title ("Distribuzione del valore di $I$ per una misura generica")
ax.hist (errori_bobina, bins = bin_edges_1, color = "thistle")
ax.set_xlabel ("intensità di corrente (A)")
ax.set_ylabel ("frequenza")

plt.show()


