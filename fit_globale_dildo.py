import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.cost import LeastSquares
from scipy.stats import chi2

# ==============================================================
# Costanti e modello
# ==============================================================

q_KT = 38.6  # q / (k_B * T) in V^-1 (temperatura ambiente)

def I_shockley(V, I_0, g):
    return I_0 * (np.exp(q_KT * V / g) - 1)

def dI_dV_shockley(V, I_0, g):
    return I_0 * np.exp(q_KT * V / g) * q_KT / g

# ==============================================================
# Dati sperimentali
# ==============================================================

# Configurazione 1: voltmetro misura V_diodo + V_amperometro
#   => V_diodo_reale = V_letto - R_A * I_letto
V_1 = np.array([0.102, 0.152, 0.173, 0.203, 0.222, 0.252, 0.272,
                0.302, 0.322, 0.352, 0.373, 0.403])  # V
I_1 = np.array([0.008, 0.035, 0.058, 0.123, 0.203, 0.431, 0.705,
                1.521, 2.550, 5.42,  9.16, 18.60])   # µA

sigma_I_conf1 = np.array([0.0205, 0.0205, 0.0205, 0.0205, 0.0205,
                           0.0205, 0.0205, 0.0205, 0.0205, 0.0202,
                           0.0202, 0.0202])  # µA

# Configurazione 2: amperometro misura I_diodo + I_voltmetro
#   => I_diodo_reale = I_letto - V_letto / R_V
V_2 = np.array([0.052, 0.072, 0.102, 0.122, 0.152, 0.173, 0.202,
                0.222, 0.252, 0.272, 0.301, 0.321, 0.351, 0.372, 0.399])  # V
I_2 = np.array([0.004, 0.009, 0.016, 0.025, 0.048, 0.073, 0.142,
                0.221, 0.447, 0.725, 1.539, 2.555, 5.410, 9.10,  18.50])  # µA

sigma_I_conf2 = np.array([0.0198, 0.0198, 0.0198, 0.0198, 0.0198,
                           0.0198, 0.0198, 0.0136, 0.0136, 0.0136,
                           0.0136, 0.0136, 0.0136, 0.0136, 0.0136])  # µA

sigma_V_1 = 0.0005  # V
sigma_V_2 = 0.0005  # V

# ==============================================================
# Modelli corretti per le due configurazioni
# ==============================================================

def I_modello_conf1(V_letto, I_0, g, R_A, R_V):
    """
    Conf. 1: il voltmetro include la caduta su R_A.
    La tensione reale sul diodo è V_letto - R_A * I_letto.
    Poiché I_letto ≈ I_diodo (R_V >> R_diodo), usiamo I_shockley
    in forma implicita; qui approssimiamo I_letto ≈ I_diodo
    (buona approssimazione se R_A piccola).
    """
    V_diodo = V_letto - R_A * I_shockley(V_letto, I_0, g) * 1e-6  # I in A
    return I_shockley(V_diodo, I_0, g)

def I_modello_conf2(V_letto, I_0, g, R_A, R_V):
    """
    Conf. 2: l'amperometro include la corrente nel voltmetro.
    La corrente reale nel diodo è I_letto - V_letto / R_V.
    """
    return I_shockley(V_letto, I_0, g) + V_letto / R_V * 1e6  # risultato in µA

# ==============================================================
# Funzione di costo globale con propagazione iterativa degli errori
# ==============================================================

def fit_globale_iterativo(n_iter=10):
    """
    Fit globale su entrambe le configurazioni con propagazione
    iterativa degli errori su V.
    """
    sigma_tot_1 = sigma_I_conf1.copy()
    sigma_tot_2 = sigma_I_conf2.copy()

    # Valori iniziali ragionevoli
    params_init = dict(I_0=0.0001, g=1.5, R_A=50.0, R_V=1e7)

    for iteration in range(n_iter):

        # Costo conf. 1
        ls1 = LeastSquares(V_1, I_1, sigma_tot_1, I_modello_conf1)
        # Costo conf. 2
        ls2 = LeastSquares(V_2, I_2, sigma_tot_2, I_modello_conf2)
        # Costo globale = somma dei due
        ls_global = ls1 + ls2

        m = Minuit(ls_global, **params_init)
        m.limits["I_0"] = (1e-8, 1.0)
        m.limits["g"]   = (1.0, 2.0)
        m.limits["R_A"] = (0.0, 1e4)    # Ohm
        m.limits["R_V"] = (1e4, 1e9)    # Ohm

        m.migrad()

        I_0_, g_, R_A_, R_V_ = (m.values["I_0"], m.values["g"],
                                 m.values["R_A"], m.values["R_V"])

        # Aggiornamento sigma_tot con propagazione errore su V
        # dI/dV calcolato sulla tensione corretta
        V_diodo_1 = V_1 - R_A_ * I_shockley(V_1, I_0_, g_) * 1e-6
        dIdV_1 = dI_dV_shockley(V_diodo_1, I_0_, g_)
        sigma_tot_1 = np.sqrt(sigma_I_conf1**2 + (dIdV_1 * sigma_V_1)**2)

        dIdV_2 = dI_dV_shockley(V_2, I_0_, g_)
        sigma_tot_2 = np.sqrt(sigma_I_conf2**2 + (dIdV_2 * sigma_V_2)**2)

        # Aggiorna i valori iniziali per la prossima iterazione
        params_init = dict(zip(m.parameters, m.values))

    # Fit finale con le sigma converge
    ls1 = LeastSquares(V_1, I_1, sigma_tot_1, I_modello_conf1)
    ls2 = LeastSquares(V_2, I_2, sigma_tot_2, I_modello_conf2)
    m_final = Minuit(ls1 + ls2, **params_init)
    m_final.limits["I_0"] = (1e-8, 1.0)
    m_final.limits["g"]   = (1.0, 2.0)
    m_final.limits["R_A"] = (0.0, 1e4)
    m_final.limits["R_V"] = (1e4, 1e9)
    m_final.migrad()
    m_final.hesse()  # errori accurati

    return m_final, sigma_tot_1, sigma_tot_2

# ==============================================================
# Esecuzione del fit
# ==============================================================

m, sigma_tot_1, sigma_tot_2 = fit_globale_iterativo(n_iter=10)

print("\n" + "="*55)
print("  RISULTATI FIT GLOBALE")
print("="*55)
for par, val, err in zip(m.parameters, m.values, m.errors):
    if par in ("R_A", "R_V"):
        print(f"  {par:5s} = {val:.2f} ± {err:.2f} Ω")
    elif par == "I_0":
        print(f"  {par:5s} = {val:.4e} ± {err:.4e} µA")
    else:
        print(f"  {par:5s} = {val:.4f} ± {err:.4f}")
print("="*55)

# ==============================================================
# Diagnostica χ²
# ==============================================================

I_0_ = m.values["I_0"]
g_   = m.values["g"]
R_A_ = m.values["R_A"]
R_V_ = m.values["R_V"]

I_pred_1 = I_modello_conf1(V_1, I_0_, g_, R_A_, R_V_)
I_pred_2 = I_modello_conf2(V_2, I_0_, g_, R_A_, R_V_)

residui_1 = (I_1 - I_pred_1) / sigma_tot_1
residui_2 = (I_2 - I_pred_2) / sigma_tot_2

chi2_tot = np.sum(residui_1**2) + np.sum(residui_2**2)
ndof = len(I_1) + len(I_2) - len(m.parameters)
p_val = chi2.sf(chi2_tot, ndof)

print(f"\n  χ²      = {chi2_tot:.2f}")
print(f"  ndof    = {ndof}")
print(f"  χ²/ndof = {chi2_tot/ndof:.2f}")
print(f"  p-value = {p_val:.4f}\n")

# ==============================================================
# Plot
# ==============================================================

x_axis = np.linspace(0.04, max(V_2) + 0.01, 5000)
y_fit_1 = I_modello_conf1(x_axis, I_0_, g_, R_A_, R_V_)
y_fit_2 = I_modello_conf2(x_axis, I_0_, g_, R_A_, R_V_)

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle("Caratterizzazione del diodo — Fit globale con correzione $R_A$, $R_V$",
             fontsize=13)

titoli = ["Scala lineare", "Scala logaritmica"]
colori = ["olivedrab", "cornflowerblue"]
etichette = ["Conf. 1 (volt. esterno)", "Conf. 2 (volt. interno)"]
V_data   = [V_1, V_2]
I_data   = [I_1, I_2]
sigma_data = [sigma_tot_1, sigma_tot_2]
y_fits   = [y_fit_1, y_fit_2]

for row, (V_d, I_d, sig, y_fit, colore, etichetta) in enumerate(
        zip(V_data, I_data, sigma_data, y_fits, colori, etichette)):

    for col, (ax, log) in enumerate(zip(axes[row], [False, True])):
        ax.set_title(f"{etichetta} — {titoli[col]}")
        ax.set_xlabel("Tensione (V)")
        ax.set_ylabel("Corrente (µA)")

        y_fit_plot = np.where(y_fit > 0, y_fit, np.nan)
        ax.plot(x_axis, y_fit_plot,
                color=colore, label="Fit globale Shockley")
        ax.errorbar(V_d, I_d,
                    xerr=sigma_V_1, yerr=sig,
                    color="indigo", linestyle="None",
                    marker="o", capsize=4, label="Dati osservati")
        if log:
            ax.set_yscale("log")
        ax.legend(fontsize=8)
        ax.grid(True, which="both", alpha=0.4)

plt.tight_layout()

# Plot residui
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4))
fig2.suptitle("Residui normalizzati — Fit globale")

for ax, res, etichetta, colore in zip(
        axes2, [residui_1, residui_2], etichette, colori):
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.bar(range(len(res)), res, color=colore, alpha=0.7)
    ax.set_title(etichetta)
    ax.set_xlabel("Indice punto")
    ax.set_ylabel("$(I_\\mathrm{obs} - I_\\mathrm{fit})\\ /\\ \\sigma$")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Plot finale di un solo unico grafico

fig, ax = plt.subplots ()

ax.set_title ("Fit globale della legge di Shockley")
ax.set_xlabel ("Tensione (V)")
ax.set_ylabel ("Intensità di corrente ($\\mu$A)")

x_axis_globale = np.linspace (min (min (V_1), min (V_2)), max (max (V_1), max (V_2)), 50000)

ax.plot (x_axis_globale,
         [I_shockley (V, m.values["I_0"], m.values["g"]) for V in x_axis_globale],
          color = "steelblue",
          label = "Legge di Shockley"
          )


ax.errorbar (V_1, I_1,
             xerr = sigma_V_1,
             yerr = sigma_tot_1,
             color = "firebrick",
             label = "Dati osservati in configurazione A",
             marker = 'o',
             linestyle = "None",
             capsize = 4
             )

ax.errorbar (V_2, I_2,
             xerr = sigma_V_2,
             yerr = sigma_tot_2,
             color = "indigo",
             marker = 'o',
             linestyle = "None",
             label = "Dati osservati in configurazione B",
             capsize = 4
             )

plt.grid (True)
plt.legend ()
plt.show ()
