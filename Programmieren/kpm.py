import numpy as np
from scipy.linalg import eigh

def kernel_polynomial_method(A, t_points, M, nvec):
    # Schritt 1: Initialisierung von ζ_k
    zeta = np.zeros(M + 1)
    
    # Schritt 2: Hauptschleife über die Anzahl der Zufallsvektoren
    for l in range(nvec):
        # Schritt 3: Auswählen eines neuen Zufallsvektors
        v_0 = np.random.randn(A.shape[0])
        v_0 /= np.linalg.norm(v_0)  # Normalisieren des Vektors
        
        # Schritt 4: Schleife über k
        v_k_minus_1 = np.zeros_like(v_0)
        v_k = v_0
        for k in range(M + 1):
            zeta[k] += np.dot(v_0, v_k)
            
            # Schritt 6: Drei-Term Rekurrenz
            if k == 0:
                v_k_plus_1 = 2 * A @ v_k
            else:
                v_k_plus_1 = 2 * A @ v_k - v_k_minus_1
            
            v_k_minus_1 = v_k
            v_k = v_k_plus_1
    
    # Schritt 8: Normierung von ζ_k
    zeta /= nvec
    
    # Schritt 9: Jackson-Dämpfung
    mu_k = zeta * (1 - (2 * np.arange(M + 1) / (M + 1))**2)
    
    # Schritt 10: Evaluierung der Dichte der Zustände (DOS)
    dos = np.zeros_like(t_points)
    for i, t in enumerate(t_points):
        T_k = np.cos(np.arange(M + 1) * np.arccos(t))
        dos[i] = np.sum(mu_k * T_k)
    
    return dos

# Beispielaufruf
n = 100
A = np.random.randn(n, n)
A = (A + A.T) / 2  # Symmetrische Matrix
t_points = np.linspace(-1, 1, 500)  # Evaluationspunkte
M = 40  # Grad des Expansionspolynoms
nvec = 10  # Anzahl der Zufallsvektoren

dos = kernel_polynomial_method(A, t_points, M, nvec)

# Plotten des Ergebnisses
import matplotlib.pyplot as plt

plt.plot(t_points, dos, label='DOS')
plt.xlabel('t')
plt.ylabel('$\phi_M(t)$')
plt.title('Approximate DOS using KPM')
plt.legend()
plt.show()