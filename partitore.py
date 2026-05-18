import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2

def V_out_partitore (R_L, R_1, R_2, V_in):
    R_eq = R_2 * R_L / (R_2 + R_L)
    return V_in * R_eq / (R_1 + R_eq)

def V_out_partitore_offset (R_L, R_1, R_2, V_in, offset):
    return V_out_partitore (R_L, R_1, R_2, V_in) + offset

if __name__ == "__main__":

    # Condizioni:
    # 1. v_out = 0.5 v_in senza considerare R_L --> R_1 = R_2
    # 2. v_out non dipende da R_L --> R_L molto grande rispetto a R_2

    R_1 = 0.67208 # k ohm
    R_2 = 0.67237 # k ohm 

    V_in = 6 # V
    V_out = np.array ([2.9058, 2.9834, 3.0010, 3.0016, 3.0022, 3.0025]) # V
    R_L = np.array ([10, 50, 300, 500, 800, 1e3]) # R_Load (k Ohm)
    sigma_V_out = np.ones (len (V_out)) * 0.001 # V

    V_out_teorico = V_out_partitore (R_L, R_1, R_2, V_in)
    offset_fit = np.average (V_out - V_out_teorico, weights = 1 / sigma_V_out**2)
    sigma_offset_fit = np.sqrt (1 / np.sum (1 / sigma_V_out**2))
    V_out_teorico_offset = V_out_partitore_offset (R_L, R_1, R_2, V_in, offset_fit)
    V_out_ideale = V_in * R_2 / (R_1 + R_2)

    chi2_partitore = np.sum (((V_out - V_out_teorico) / sigma_V_out) ** 2)
    ndof_partitore = len (V_out)
    p_value_partitore = chi2.sf (chi2_partitore, ndof_partitore)

    chi2_partitore_offset = np.sum (((V_out - V_out_teorico_offset) / sigma_V_out) ** 2)
    ndof_partitore_offset = len (V_out) - 1
    p_value_partitore_offset = chi2.sf (chi2_partitore_offset, ndof_partitore_offset)

    print (f"V_out ideale senza carico = {V_out_ideale:.4f} V")
    print (f"Chi2 partitore = {chi2_partitore:.2f}")
    print (f"ndof partitore = {ndof_partitore}")
    print (f"Chi2 ridotto partitore = {chi2_partitore / ndof_partitore:.2f}")
    print (f"p-value partitore = {p_value_partitore:.4f}")
    print (f"Offset = {offset_fit:.4f} \\pm {sigma_offset_fit:.4f} V")
    print (f"Chi2 partitore con offset = {chi2_partitore_offset:.2f}")
    print (f"ndof partitore con offset = {ndof_partitore_offset}")
    print (f"Chi2 ridotto partitore con offset = {chi2_partitore_offset / ndof_partitore_offset:.2f}")
    print (f"p-value partitore con offset = {p_value_partitore_offset:.4f}")

    # grafico

    fig, ax = plt.subplots ()
    R_L_plot = np.logspace (np.log10 (min (R_L)), np.log10 (max (R_L)), 500)

    ax.set_title ("Funzionamento di un partitore resistivo")
    ax.set_xlabel ("R$_{load}$ (k$\\Omega$)")
    ax.set_ylabel ("V$_{out}$ (V)")
    #ax.set_xscale ("log")

    ax.errorbar (R_L, V_out,
                 yerr = sigma_V_out,
                 color = "indigo",
                 linestyle = "None",
                 marker = "o",
                 capsize = 4,
                 label = "Dati osservati"
                 )

    ax.plot (R_L_plot,
             V_out_partitore (R_L_plot, R_1, R_2, V_in),
             color = "cornflowerblue",
             label = "Partitore caricato"
             )

    ax.plot (R_L_plot,
             V_out_partitore_offset (R_L_plot, R_1, R_2, V_in, offset_fit),
             color = "tomato",
             label = "Partitore caricato con offset"
             )
    
    ax.axhline (y = V_out_ideale,
                linestyle = "--",
                color = "crimson",
                label = "Limite senza carico"
                )
    
    ax.legend ()
    ax.grid (True, which = "both")
    plt.tight_layout ()
    plt.show ()
