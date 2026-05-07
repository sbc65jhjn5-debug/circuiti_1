import numpy as np
import matplotlib.pyplot as plt

with open ("errori_1.txt", "r") as f:
    errori_1 = np.array ([float (x) for x in f.read ().split ()])

I_medio = np.mean (errori_1)
I_std = np.std (errori_1, ddof = 1)
print (f"Valore medio di I: {I_medio:.5f} mA")
print (f"Deviazione standard di I (errore da utilizzare per misure con mA): {I_std:.5f} mA")

n_bins_1 = 15
bin_edges_1 = np.linspace (min (errori_1), max (errori_1), n_bins_1 + 1)

fig, ax = plt.subplots ()

ax.set_title ("Distribuzione del valore di $I$ per una misura generica")
ax.hist (errori_1, bins = bin_edges_1, color = "lightblue")
ax.set_xlabel ("intensità di corrente (mA)")
ax.set_ylabel ("frequenza")

plt.show ()



