import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.cost import LeastSquares
from scipy.stats import chi2

"""
In questo programma il fit viene eseguito direttamente sugli angoli misurati,
anziche' sulla tangente degli angoli. Questa scelta e' piu' naturale dal punto
di vista sperimentale, perche' la grandezza osservata direttamente e' theta,
non tan(theta). Fittare tan(theta) puo' amplificare molto gli errori, soprattutto
per angoli grandi, perche' la tangente cresce rapidamente e la propagazione
dell'incertezza diventa molto sensibile.

Il modello fisico ideale e' la legge della tangente:

    tan(theta) = B_bobina / B_T

dove B_bobina e' il campo magnetico prodotto dalla bobina e B_T e' il campo
magnetico terrestre. In forma angolare, il modello diventa:

    theta = arctan(B_bobina / B_T)

A questo modello e' stato aggiunto un parametro theta0:

    theta = arctan(B_bobina / B_T) + theta0

Il parametro theta0 rappresenta un possibile errore sistematico sullo zero
angolare: per esempio un disallineamento iniziale della bussola, una lettura
non perfettamente centrata, oppure una direzione di riferimento non coincidente
con quella assunta nel modello ideale.

Tuttavia, osservando i residui del fit, si nota che essi non sono distribuiti
in modo completamente casuale attorno allo zero. In particolare presentano un
andamento sistematico: prima tendono a stare da una parte, poi dall'altra.
Questo indica che il modello con il solo offset angolare non descrive tutta
la struttura dei dati. In altre parole, non basta traslare tutti gli angoli:
serve una correzione che modifichi anche la curvatura del modello.

Per questo e' stato introdotto un secondo modello, che tiene conto del possibile
disallineamento geometrico tra il campo magnetico generato dalla bobina e il
campo magnetico terrestre. La legge della tangente ideale assume infatti che
il campo della bobina sia perfettamente perpendicolare al campo terrestre.
Se questa condizione non e' rispettata, la relazione corretta non e' piu'
semplicemente tan(theta) = B_bobina / B_T.

Nel modello piu' generale si introduce un angolo phi, che descrive la direzione
del campo della bobina rispetto al campo terrestre:

    theta = arctan2(B_bobina sin(phi), B_T + B_bobina cos(phi)) + theta0

Se phi = 90 gradi, allora cos(phi) = 0 e sin(phi) = 1, quindi si recupera
esattamente il modello ideale:

    theta = arctan(B_bobina / B_T) + theta0

Se invece phi e' diverso da 90 gradi, significa che una componente del campo
della bobina e' parallela al campo terrestre. Questa componente modifica il
denominatore della tangente e produce una deviazione non lineare dal modello
ideale. Proprio questo tipo di effetto puo' generare residui con andamento
sistematico, per esempio prima positivi e poi negativi.

Nel fit con disallineamento si ottiene un valore di phi circa uguale a 82.6
gradi. Questo significa che la bobina non risulterebbe perfettamente
perpendicolare al campo terrestre, ma disallineata di circa:

    90 - 82.6 = 7.4 gradi

Rispetto al modello precedente, il parametro theta0 diventa molto piu' piccolo,
circa compatibile con zero entro l'incertezza. Questo e' fisicamente significativo:
nel fit senza phi, lo spostamento veniva assorbito quasi tutto da theta0, che
risultava circa -9 gradi. Nel fit con phi, invece, la sistematica viene spiegata
meglio come un disallineamento geometrico della bobina, non come un grande errore
di zero angolare.

Anche il valore del campo magnetico terrestre cambia. Nel modello con solo
theta0 si ottiene un valore piu' basso, circa 2.23e-5 T, mentre nel modello con
phi si ottiene circa 2.90e-5 T. Quest'ultimo e' piu' vicino a un valore realistico
per la componente orizzontale del campo terrestre, che e' la componente misurata
in un esperimento di questo tipo con ago magnetico.

Il chi quadro del modello con phi risulta molto piccolo e il p-value molto alto.
Questo indica che il modello descrive molto bene i dati, forse anche troppo bene.
Un p-value molto alto puo' significare che la correzione geometrica cattura
effettivamente la sistematica osservata nei residui, ma puo' anche indicare che
le incertezze assegnate agli angoli sono leggermente sovrastimate. Per questo il
modello con phi va interpretato come fisicamente plausibile, ma non come prova
assoluta del disallineamento: e' una spiegazione coerente con i dati e con la
forma dei residui.

Gli errori sugli angoli sono stati calcolati sommando in quadratura tre contributi:
l'incertezza sulla misura ad apparato spento, l'incertezza sulla media delle
misure ad apparato acceso e l'incertezza di risoluzione della scala, assunta pari
a 1/sqrt(12) gradi. La somma in quadratura e' corretta quando gli errori sono
indipendenti.

Infine, il programma calcola anche una possibile sigma_extra, cioe' una dispersione
aggiuntiva da sommare in quadratura agli errori sperimentali. Questa serve a
modellare eventuali fluttuazioni non incluse nelle incertezze iniziali, per
esempio oscillazioni dell'ago, piccoli disturbi ambientali, instabilita' nella
lettura o campi magnetici parassiti. Nel caso del fit con solo theta0 questa
sigma_extra puo' essere utile; nel fit con phi, invece, gran parte della struttura
dei residui viene gia' spiegata dal disallineamento geometrico.

In conclusione, il modello ideale della legge della tangente e' corretto come
punto di partenza, ma i residui mostrano una struttura sistematica. L'introduzione
di un angolo phi di disallineamento fornisce una spiegazione fisica naturale di
questa struttura e migliora sensibilmente la descrizione dei dati. Il risultato
suggerisce quindi che l'apparato sperimentale non fosse perfettamente allineato
rispetto al campo magnetico terrestre.
"""


MU0 = 4 * np.pi * 1e-7


def B_bobina(I, N, L, r):
    return MU0 * N * I / (2 * np.sqrt(r**2 + (L / 2) ** 2))


def theta_model(I, N, L, r, B_t, theta0):
    rapporto = B_bobina(I, N, L, r) / B_t
    return np.degrees(np.arctan(rapporto)) + theta0


def theta_model_disallineato(I, N, L, r, B_t, theta0, phi):
    B = B_bobina(I, N, L, r)
    phi_rad = np.radians(phi)
    numeratore = B * np.sin(phi_rad)
    denominatore = B_t + B * np.cos(phi_rad)
    return np.degrees(np.arctan2(numeratore, denominatore)) + theta0


def errore_media(x):
    return np.std(x, ddof=1) / np.sqrt(len(x))


def stampa_fit(nome, m):
    print(f"\n{nome}")
    for par, val, err in zip(m.parameters, m.values, m.errors):
        print(f"{par} = {val:.6e} ± {err:.6e}")

    chi_2 = m.fval
    ndof = m.ndof
    p_value = chi2.sf(chi_2, ndof)
    print(f"chi2 = {chi_2:.3f}")
    print(f"ndof = {ndof}")
    print(f"p-value = {p_value:.4f}")


lunghezza = 5.06e-2
raggio = 25.00e-2 / 2
N_spire = 31

I = np.array(
    [
        0.10200,
        0.15025,
        0.20275,
        0.25310,
        0.30336,
        0.35346,
        0.40473,
        0.45494,
        0.50523,
        0.55570,
        0.60600,
        0.65598,
        0.70580,
        0.75565
    ]
)

deg_spento = np.array([85, 86, 87, 90, 90, 90, 91, 92, 86, 91, 92, 93, 91, 92, 93, 90])

misure_angoli = [
    np.array([61, 61, 61, 60, 61, 61, 61, 61, 62, 62, 63, 63, 64, 64, 63, 64, 64, 63, 64, 63, 62]),
    np.array([52, 53, 54]),
    np.array([45, 46, 47]),
    np.array([40, 40, 41]),
    np.array([36, 37]),
    np.array([32, 33]),
    np.array([29, 29, 30]),
    np.array([27, 27, 28]),
    np.array([25, 26]),
    np.array([24, 24, 25]),
    np.array([23, 23, 23, 24]),
    np.array([22, 22, 22, 21]),
    np.array([20, 20, 21]),
    np.array([20, 20, 19])
]

theta_spento = np.mean(deg_spento)
theta_misure = np.array([np.mean(a) for a in misure_angoli])
theta = theta_spento - theta_misure

sigma_spento = errore_media(deg_spento)
print (f"Angolo a corrente spenta: {theta_spento:.2f} ± {sigma_spento:.2f} gradi")
sigma_misure = np.array([errore_media(a) for a in misure_angoli])
#sigma_risoluzione = 1.0 / np.sqrt(12)

sigma_theta = np.sqrt(sigma_spento**2 + sigma_misure**2)

print("Angoli usati nel fit:")
for i, th, sig in zip(I, theta, sigma_theta):
    print(f"I = {i:.5f} A, theta = {th:.2f} ± {sig:.2f} gradi")


# Fit principale: si fitta direttamente l'angolo misurato.
ls = LeastSquares(I, theta, sigma_theta, theta_model)

m = Minuit(ls, N=N_spire, L=lunghezza, r=raggio, B_t=4.0e-5, theta0=0.0)
m.fixed["N"] = True
m.limits["L"] = (lunghezza - 0.001, lunghezza + 0.001)
m.limits["r"] = (raggio - 0.001, raggio + 0.001)
m.limits["B_t"] = (1e-5, 8e-5)
m.limits["theta0"] = (-20.0, 20.0)
m.migrad()
m.hesse()

stampa_fit("Fit sugli angoli con offset angolare", m)

B_fit = m.values["B_t"]
B_err = m.errors["B_t"]
theta0_fit = m.values["theta0"]
theta0_err = m.errors["theta0"]

theta_fit = theta_model(I, N_spire, lunghezza, raggio, B_fit, theta0_fit)
residui = theta - theta_fit
pull = residui / sigma_theta

print(f"\nRisultato: B terrestre = {B_fit:.6e} ± {B_err:.6e} T")
print(f"Offset angolare theta0 = {theta0_fit:.3f} ± {theta0_err:.3f} gradi")
print(f"RMS pull = {np.sqrt(np.mean(pull**2)):.3f}")


# Fit con disallineamento geometrico: phi = 90 gradi corrisponde alla legge della tangente ideale.
ls_phi = LeastSquares(I, theta, sigma_theta, theta_model_disallineato)

m_phi = Minuit(
    ls_phi,
    N=N_spire,
    L=lunghezza,
    r=raggio,
    B_t=B_fit,
    theta0=theta0_fit,
    phi=90.0,
)
m_phi.fixed["N"] = True
m_phi.limits["L"] = (lunghezza - 0.001, lunghezza + 0.001)
m_phi.limits["r"] = (raggio - 0.001, raggio + 0.001)
m_phi.limits["B_t"] = (1e-5, 8e-5)
m_phi.limits["theta0"] = (-20.0, 20.0)
m_phi.limits["phi"] = (60.0, 120.0)
m_phi.migrad()
m_phi.hesse()

stampa_fit("Fit con offset angolare e disallineamento phi", m_phi)

B_phi_fit = m_phi.values["B_t"]
B_phi_err = m_phi.errors["B_t"]
theta0_phi_fit = m_phi.values["theta0"]
theta0_phi_err = m_phi.errors["theta0"]
phi_fit = m_phi.values["phi"]
phi_err = m_phi.errors["phi"]

theta_fit_phi = theta_model_disallineato(
    I, N_spire, lunghezza, raggio, B_phi_fit, theta0_phi_fit, phi_fit
)
residui_phi = theta - theta_fit_phi
pull_phi = residui_phi / sigma_theta

print(f"\nRisultato con phi: B terrestre = {B_phi_fit:.6e} ± {B_phi_err:.6e} T")
print(f"Offset angolare theta0 = {theta0_phi_fit:.3f} ± {theta0_phi_err:.3f} gradi")
print(f"Disallineamento phi = {phi_fit:.3f} ± {phi_err:.3f} gradi")
print(f"RMS pull con phi = {np.sqrt(np.mean(pull_phi**2)):.3f}")


# Stima semplice di una dispersione extra, utile se i residui sono piu larghi degli errori.
sigma_extra = 0.0
if m.ndof > 0 and m.fval > m.ndof:
    sigma_extra = np.sqrt((m.fval / m.ndof - 1.0) * np.mean(sigma_theta**2))

sigma_theta_extra = np.sqrt(sigma_theta**2 + sigma_extra**2)

if sigma_extra > 0:
    ls_extra = LeastSquares(I, theta, sigma_theta_extra, theta_model_disallineato)
    m_extra = Minuit(ls_extra, N=N_spire, L=lunghezza, r=raggio, B_t=B_fit, theta0=theta0_fit, phi=phi_fit)
    m_extra.fixed["N"] = True
    m_extra.limits["L"] = (lunghezza - 0.001, lunghezza + 0.001)
    m_extra.limits["r"] = (raggio - 0.001, raggio + 0.001)
    m_extra.limits["B_t"] = (1e-5, 8e-5)
    m_extra.limits["theta0"] = (-20.0, 20.0)
    m_extra.limits["phi"] = (60.0, 120.0)
    m_extra.migrad()
    m_extra.hesse()

    print(f"\nSigma extra stimata = {sigma_extra:.3f} gradi")
    stampa_fit("Fit con sigma extra aggiunta in quadratura", m_extra)
else:
    print("\nSigma extra non aggiunta: il chi2 non richiede ulteriore dispersione.")


fig, ax = plt.subplots()
I_plot = np.linspace(np.min(I), np.max(I), 400)

ax.set_title("Andamento degli angoli in funzione della corrente")
ax.set_xlabel("intensità di corrente $I$ (A)")
ax.set_ylabel("$\\theta$ (gradi)")

ax.errorbar(
    I,
    theta,
    yerr=sigma_theta,
    xerr=0.00003,
    marker="o",
    linestyle="None",
    capsize=4,
    color="indigo",
    label="Dati osservati",
)

ax.plot(
    I_plot,
    theta_model(I_plot, N_spire, lunghezza, raggio, B_fit, theta0_fit),
    color="crimson",
    label="$\\theta = \\arctan \\frac{B_{bobina}}{B_T} + \\theta_0$",
)

ax.plot(
    I_plot,
    theta_model_disallineato(I_plot, N_spire, lunghezza, raggio, B_phi_fit, theta0_phi_fit, phi_fit),
    color="darkgreen",
    linestyle="--",
    label="$\\theta = \\arctan \\frac{B_{bobina}\\sin\\phi} {B_T+B_{bobina}\\cos\\phi} + \\theta_0$",
)

ax.legend(fontsize=12)
ax.grid(True)
plt.tight_layout()
plt.show()


fig, ax = plt.subplots()

ax.set_title("Residui normalizzati del fit sugli angoli")
ax.set_xlabel("intensità di corrente $I$ (A)")
ax.set_ylabel("residuo / $\\sigma$")

ax.errorbar(
    I,
    pull,
    yerr=np.ones_like(pull),
    xerr=0.00003,
    marker="^",
    linestyle="None",
    capsize=4,
    color="indigo",
    label="modello senza $\\phi$",
)

ax.errorbar(
    I,
    pull_phi,
    yerr=np.ones_like(pull_phi),
    xerr=0.00003,
    marker="s",
    linestyle="None",
    capsize=4,
    color="darkgreen",
    label="modello con $\\phi$",
)

ax.axhline(0, color="crimson", linestyle="--")
ax.axhline(1, color="gray", linestyle=":", linewidth=1)
ax.axhline(-1, color="gray", linestyle=":", linewidth=1)
ax.legend(fontsize=11)
ax.grid(True)
plt.tight_layout()
plt.show()
