import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # Condizioni:
    # 1. v_out = 0.5 v_in senza considerare R_L --> R_1 = R_2
    # 2. v_out non dipende da R_L --> R_L molto grande rispetto a R_2

    R_1 = 672.08 # ohm
    R_2 = 672.37 # ohm 

    V_in = 6 # V
    V_out = np.array ([2.9058, 2.9834, 3.0034, 3.0010, 3.0016, 3.0022, 3.0025]) # V
    R_L = np.array ([10e3, 50e3, 100e3, 300e3, 500e3, 800e3, 1e6]) # R_Load (ohm)

    # grafico

    fig, ax = plt.subplots ()

    ax.set_title ("FUnzionamento di un partitore resistivo")
    ax.set_xlabel ("R$_{load}$ ($\\Omega$)")
    ax.set_ylabel ("V$_{out}$ (V)")

    ax.errorbar (R_L, V_out,
                 yerr = 0.001,
                 color = "indigo",
                 linestyle = "None",
                 marker = "o",
                 capsize = 4
                 )
    
    ax.axhline (y = 3.00,
                linestyle = "--",
                color = "crimson"
                )
    
    plt.grid (True)
    plt.show ()
