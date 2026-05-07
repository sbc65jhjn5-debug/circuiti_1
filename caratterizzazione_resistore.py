import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.cost import LeastSquares

if __name__ == "__main__":

    # Configurazione 1:
    #                 ----- amperometro --- resistenza ----
    # generatore ----                                       -----
    #                 -------------- voltmetro ------------

    # Configurazione 2:
    #                                  ----- resistenza -----
    # generatore ---- amperometro ----                        ----
    #                                  ----- voltmetro ------

    def I (x, R):
        return x / R
    
    R_1 = 100 # ohm -----> video 1
    R_2 = 20e3 # ohm ----> video 1
    R_3 = 3e6 # ohm -----> video 3

    # generate sempre con .00

    # Configurazione 1, R 1 (buona per stima R_amperometro)

    voltmetro_1_1 = np.array ([0.102, 0.201, 0.302, 0.403, 0.503, 0.702, 0.804, 1.003, 1.202]) # V
    amperometro_1_1 = np.array([0.981, 1.946, 2.913, 3.872, 4.834, 6.754 ,7.729, 9.654, 11.574]) # mA
    sigma_1_1 = np.array (0.0012 * np.ones (len (voltmetro_1_1))) # V 

    # Configurazione 1, R 2

    voltmetro_2_1 = np.array ([0.102, 1.003, 1.502, 2.003, 2.503, 3.004, 3.504, 4.004, 4.504]) # V
    amperometro_2_1 = np.array ([0.005, 0.050, 0.075, 0.100, 0.125, 0.151, 0.176, 0.201, 0.226]) # mA
    sigma_2_1 = np.array (0.0012 * np.ones (len (voltmetro_2_1))) # V 
    
    # Configurazione 1, R 3

    voltmetro_3_1 = np.array ([3.004, 5.205, 8.01, 11.21, 14.01, 17.01])
    amperometro_3_1 = np.array ([0.001, 0.002, 0.003, 0.004, 0.005, 0.006]) # mA
    sigma_3_1 = np.array (0.0012 * np.ones (len (voltmetro_3_1))) # V (da mettere del micro...)

    # Configurazione 2, R 1 (buona per stima R1)

    voltmetro_1_2 = np.array ([0.100, 0.197, 0.295, 0.393, 0.490, 0.588, 0.686, 0.784, 0.881, 0.979])
    amperometro_1_2 = np.array ([0.979, 1.943, 2.907, 3.883, 4.846, 5.811, 6.775, 7.750, 8.714, 9.679]) # mA
    sigma_1_2 = np.array (0.0012 * np.ones (len (voltmetro_1_2))) # V 
    
    # Configurazione 2, R 2

    voltmetro_2_2 = np.array ([0.503, 1.003, 1.502, 2.002, 2.502, 3.004, 3.504, 4.003])
    amperometro_2_2 = np.array ([0.025, 0.050, 0.075, 0.100, 0.126, 0.151, 0.176, 0.201])
    sigma_2_2 = np.array (0.0012 * np.ones (len (voltmetro_2_2))) # V 

    # Configurazione 2, R 3

    voltmetro_3_2 = np.array ([3.004, 4.204, 6.504, 8.91, 11.01, 13.32, 15.72])
    amperometro_3_2 = np.array ([0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007])
    sigma_3_2 = np.array (0.0012 * np.ones (len (voltmetro_3_2))) # V (da mettere del micro...)

    # Fit per la resistenza R 1

    ls1 = LeastSquares (voltmetro_1_1,
                       amperometro_1_1,
                       sigma_1_1, 
                       I
                       )
    
    m1 = Minuit (ls1, 
                R = R_1 * 1e-3 # kΩ
                )
    

    m1.migrad ()

    for par, val, err in zip (m1.parameters, m1.values, m1.errors):
        print (f"{par} = {val:.3f} ± {err:.3f}")
    
    R_fit1 = m1.values["R"]

    ls2 = LeastSquares (voltmetro_1_2,
                       amperometro_1_2,
                       sigma_1_2,
                       I
                       )
    
    m2 = Minuit (ls2,
                R = R_1 * 1e-3 # kΩ
                )
    
    m2.migrad ()

    for par, val, err in zip (m2.parameters, m2.values, m2.errors):
        print (f"{par} = {val:.3f} ± {err:.3f}")

    R_fit2 = m2.values["R"]

    # plot  per la resistenza R 1

    fig, ax = plt.subplots (nrows = 2, ncols = 1)

    ax[0].set_title ("Andamento dell'intensità di corrente in funzione della tensione")
    ax[0].set_xlabel ("tensione (V)")
    ax[0].set_ylabel ("intensità di corrente (mA)")
    ax[0].errorbar (voltmetro_1_1, amperometro_1_1, yerr = sigma_1_1, capsize = 4, fmt = "o", linestyle = "None", color = "mediumseagreen", label = "configurazione 1")
    ax[0].plot (voltmetro_1_1, I (voltmetro_1_1, R_fit1), "-", color = "mediumseagreen", label = f"fit configurazione 1, R = {R_fit1:.3f} kΩ")
    ax[0].legend () 
    ax[0].grid ()
    ax[1].errorbar (voltmetro_1_2, amperometro_1_2, yerr = sigma_1_2, capsize = 4, fmt = "o", linestyle = "None", color = "lightcoral", label = "configurazione 2")
    ax[1].plot (voltmetro_1_2, I (voltmetro_1_2, R_fit2), "-", color = "lightcoral", label = f"fit configurazione 2, R = {R_fit2:.3f} kΩ")
    ax[1].set_xlabel ("tensione (V)")
    ax[1].set_ylabel ("intensità di corrente (mA)")
    ax[1].legend ()
    ax[1].grid ()
    plt.show ()


    # Fit per la resistenza R 2
    
    ls21 = LeastSquares (voltmetro_2_1,
                       amperometro_2_1,
                       sigma_2_1, 
                       I
                       )
    
    m21 = Minuit (ls21, 
                R = R_2 * 1e-3 # kΩ
                )
    

    m21.migrad ()

    for par, val, err in zip (m21.parameters, m21.values, m21.errors):
        print (f"{par} = {val:.3f} ± {err:.3f}")
    
    R_fit21 = m21.values["R"]

    ls22 = LeastSquares (voltmetro_2_2,
                       amperometro_2_2,
                       sigma_2_2,
                       I
                       )
    
    m22 = Minuit (ls22,
                R = R_2 * 1e-3 # kΩ
                )
    
    m22.migrad ()

    for par, val, err in zip (m22.parameters, m22.values, m22.errors):
        print (f"{par} = {val:.3f} ± {err:.3f}")

    R_fit22 = m22.values["R"]

    # plot  per la resistenza R 2

    fig, ax = plt.subplots (nrows = 2, ncols = 1)

    ax[0].set_title ("Andamento dell'intensità di corrente in funzione della tensione")
    ax[0].set_xlabel ("tensione (V)")
    ax[0].set_ylabel ("intensità di corrente (mA)")
    ax[0].errorbar (voltmetro_2_1, amperometro_2_1, yerr = sigma_2_1, capsize = 5, fmt = "o", linestyle = "None", color = "seagreen", label = "configurazione 1")
    ax[0].plot (voltmetro_2_1, I (voltmetro_2_1, R_fit21), "-", color = "seagreen", label = f"fit configurazione 1, R = {R_fit21:.3f} kΩ")
    ax[0].legend () 
    ax[0].grid ()
    ax[1].errorbar (voltmetro_2_2, amperometro_2_2, yerr = sigma_2_2, capsize = 5, fmt = "o", linestyle = "None", color = "indianred", label = "configurazione 2")
    ax[1].plot (voltmetro_2_2, I (voltmetro_2_2, R_fit22), "-", color = "indianred", label = f"fit configurazione 2, R = {R_fit22:.3f} kΩ")
    ax[1].set_xlabel ("tensione (V)")
    ax[1].set_ylabel ("intensità di corrente (mA)")
    ax[1].legend ()
    ax[1].grid ()
    plt.show ()

   # Fit per la resisteenza R 3 
    
    ls31 = LeastSquares (voltmetro_3_1,
                       amperometro_3_1,
                       sigma_3_1, 
                       I
                       )
    
    m31 = Minuit (ls31, 
                R = R_3 * 1e-3 # kΩ
                )
    

    m31.migrad ()

    for par, val, err in zip (m31.parameters, m31.values, m31.errors):
        print (f"{par} = {val:.3f} ± {err:.3f}")
    
    R_fit31 = m31.values["R"]

    ls32 = LeastSquares (voltmetro_3_2,
                       amperometro_3_2,
                       sigma_3_2,
                       I
                       )
    
    m32 = Minuit (ls32,
                R = R_3 * 1e-3 # kΩ
                )
    
    m32.migrad ()

    for par, val, err in zip (m32.parameters, m32.values, m32.errors):
        print (f"{par} = {val:.3f} ± {err:.3f}")

    R_fit32 = m32.values["R"]

    # plot  per la resistenza R 3   

    fig, ax = plt.subplots (nrows = 2, ncols = 1)

    ax[0].set_title ("Andamento dell'intensità di corrente in funzione della tensione")
    ax[0].set_xlabel ("tensione (V)")
    ax[0].set_ylabel ("intensità di corrente (mA)")
    ax[0].errorbar (voltmetro_3_1, amperometro_3_1, yerr = sigma_3_1, capsize = 5, fmt = "o", linestyle = "None", color = "darkgreen", label = "configurazione 1")
    ax[0].plot (voltmetro_3_1, I (voltmetro_3_1, R_fit31), "-", color = "darkgreen", label = f"fit configurazione 1, R = {R_fit31:.3f} kΩ")
    ax[0].legend () 
    ax[0].grid ()
    ax[1].errorbar (voltmetro_3_2, amperometro_3_2, yerr = sigma_3_2, capsize = 5, fmt = "o", linestyle = "None", color = "brown", label = "configurazione 2")
    ax[1].plot (voltmetro_3_2, I (voltmetro_3_2, R_fit32), "-", color = "brown", label = f"fit configurazione 2, R = {R_fit32:.3f} kΩ")
    ax[1].set_xlabel ("tensione (V)")
    ax[1].set_ylabel ("intensità di corrente (mA)")
    ax[1].legend ()
    ax[1].grid ()
    plt.show ()