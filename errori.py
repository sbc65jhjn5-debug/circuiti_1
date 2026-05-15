import numpy as np
import matplotlib.pyplot as plt

with open ("errori_milli.txt", "r") as f:
    errori_milli = np.array ([float (x) for x in f.read ().split ()])

I_medio = np.mean (errori_milli)
I_std = np.std (errori_milli, ddof = 1)
print (f"Valore medio di I: {I_medio:.5f} mA")
print (f"Deviazione standard di I (errore da utilizzare per misure con mA): {I_std:.5f} mA")

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
print (f"Deviazione standard di I (errore da utilizzare per misure con $\\mu$A): {I_std:.5f} $\\mu$A")

n_bins_1 = 12
bin_edges_1 = np.linspace (min (errori_micro), max (errori_micro), n_bins_1 + 1)

fig, ax = plt.subplots ()

ax.set_title ("Distribuzione del valore di $I$ per una misura generica")
ax.hist (errori_micro, bins = bin_edges_1, color = "bisque")
ax.set_xlabel ("intensità di corrente ($\\mu$A)")
ax.set_ylabel ("frequenza")

plt.show()

# errori dildo 1

with open ("errori_dildo1.txt", "r") as f:
    errori_dildo1 = np.array ([float (x) for x in f.read ().split ()])

I_medio = np.mean (errori_dildo1)
I_std = np.std (errori_dildo1, ddof = 1)
print (f"Valore medio di I: {I_medio:.5f} $\\mu$A")
print (f"Deviazione standard di I (errore da utilizzare per misure con $\\mu$A): {I_std:.5f} $\\mu$A")

n_bins_1 = 8
bin_edges_1 = np.linspace (min (errori_dildo1), max (errori_dildo1), n_bins_1 + 1)

fig, ax = plt.subplots ()

ax.set_title ("Distribuzione del valore di $I$ per una misura generica")
ax.hist (errori_dildo1, bins = bin_edges_1, color = "pink")
ax.set_xlabel ("intensità di corrente ($\\mu$A)")
ax.set_ylabel ("frequenza")

plt.show()

# errori dildo 2

with open ("errori_dildo_2.txt", "r") as f:
    errori_dildo2 = np.array ([float (x) for x in f.read ().split ()])

I_medio = np.mean (errori_dildo2)
I_std = np.std (errori_dildo2, ddof = 1)
print (f"Valore medio di I: {I_medio:.5f} $\\mu$A")
print (f"Deviazione standard di I (errore da utilizzare per misure con $\\mu$A): {I_std:.5f} $\\mu$A")

n_bins_1 = 10
bin_edges_1 = np.linspace (min (errori_dildo2), max (errori_dildo2), n_bins_1 + 1)

fig, ax = plt.subplots ()

ax.set_title ("Distribuzione del valore di $I$ per una misura generica")
ax.hist (errori_dildo2, bins = bin_edges_1, color = "lemonchiffon")
ax.set_xlabel ("intensità di corrente ($\\mu$A)")
ax.set_ylabel ("frequenza")

plt.show()

# errori corrente bobina

with open ("errori_corrente_bobina.txt", "r") as f:
    errori_bobina = np.array ([float (x) for x in f.read ().split ()])

I_medio = np.mean (errori_bobina)
I_std = np.std (errori_bobina, ddof = 1)
print (f"Valore medio di I: {I_medio:.5f} $\\mu$A")
print (f"Deviazione standard di I (errore da utilizzare per misure con $\\mu$A): {I_std:.5f} $\\mu$A")

n_bins_1 = 9
bin_edges_1 = np.linspace (min (errori_bobina), max (errori_bobina), n_bins_1 + 1)

fig, ax = plt.subplots ()

ax.set_title ("Distribuzione del valore di $I$ per una misura generica")
ax.hist (errori_bobina, bins = bin_edges_1, color = "thistle")
ax.set_xlabel ("intensità di corrente ($\\mu$A)")
ax.set_ylabel ("frequenza")

plt.show()


