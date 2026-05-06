import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # Configurazione 1:
    #                 ----- amperometro --- resistenza ----
    # generatore ----                                       -----
    #                 -------------- voltmetro ------------

    # Configurazione 2:
    #                                  ----- resistenza -----
    # generatore ---- amperometro ----                        ----
    #                                  ----- voltmetro ------

    R_1 = 100 # ohm -----> video 1
    R_2 = 20e3 # ohm ----> video 1
    R_3 = 3e6 # ohm -----> video 3

    # generate sempre con .00

    # Configurazione 1, R 1 (buona per stima R_amperometro)

    voltmetro_1_1 = np.array ([0.102, 0.201, 0.302, 0.403, 0.503, 0.702, 0.804, 1.003, 1.202]) # V
    amperometro_1_1 = np.array([0.981, 1.946, 2.913, 3.872, 4.834, 6.754 ,7.729, 9.654, 11.574]) # mA

    # Configurazione 1, R 2

    voltmetro_2_1 = np.array ([0.102, 1.003, 1.502, 2.003, 2.503, 3.004, 3.504, 4.004, 4.504]) # V
    amperometro_2_1 = np.array ([0.005, 0.050, 0.075, 0.100, 0.125, 0.151, 0.176, 0.201, 0.226]) # mA

    # Configurazione 1, R 3

    voltmetro_3_1 = np.array ([3.004, 5.205, 8.01, 11.21, 14.01, 17.01])
    amperometro_3_1 = np.array ([0.001, 0.002, 0.003, 0.004, 0.005, 0.006])

    # Configurazione 2, R 1 (buona per stima R1)

    voltmetro_1_2 = np.array ([0.100, 0.197, 0.295, 0.393, 0.490, 0.588, 0.686, 0.784, 0.881, 0.979])
    amperometro_1_2 = np.array ([0.979, 1.943, 2.907, 3.883, 4.846, 5.811, 6.775, 7.750, 8.714, 9.679])

    # Configurazione 2, R 2

    voltmetro_2_2 = np.array ([0.503, 1.003, 1.502, 2.002, 2.502, 3.004, 3.504, 4.003])
    amperometro_2_2 = np.array ([0.025, 0.050, 0.075, 0.100, 0.126, 0.151, 0.176, 0.201])

    # Configurazione 2, R 3

    voltmetro_3_2 = np.array ([3.004, 4.204, 6.504, 8.91, 11.01, 13.32, 15.72])
    amperometro_3_2 = np.array ([0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007])

    # plot  per la resistenza R 1

    fig, ax = plt.subplots (nrows = 2, ncols = 1)

    ax.set_title ("Andamento dell'intensità di corrente in funzione della tensione")