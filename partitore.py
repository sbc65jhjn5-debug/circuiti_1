import numpy as np

if __name__ == "__main___":

    # Condizioni:
    # 1. v_out = 0.5 v_in senza considerare R_L --> R_1 = R_2
    # 2. v_out non dipende da R_L --> R_L molto grande rispetto a R_2

    R_1 = 672.08 # ohm
    R_2 = 672.37 # ohm 

    V_in = 6 # V
    V_out = np.array ([2.9058, 2.9834, 3.0034, 3.0010, 3.0016, 3.0022, 3.0025]) # V
    R_L = np.array ([10e3, 50e3, 100e3, 300e3, 500e3, 800e3, 1e6]) # R_Load (ohm)