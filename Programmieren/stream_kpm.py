import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import pi, sqrt

def generate_symmetric_matrix(n):
    A = np.random.randn(n, n)
    return (A + A.T)/2

def shift_eigenvalues(A):
    eigenvalues, _ = np.linalg.eigh(A)
    c = (max(eigenvalues) + min(eigenvalues))/2
    d = (max(eigenvalues) - min(eigenvalues))/2
    B = (A - c * np.identity(len(A)))/d
    return B

def generate_dos_plot(t_points, dos):
    fig, ax = plt.subplots()
    ax.plot(st.session_state["t_points"][1:-1], st.session_state["dos"], label='DOS')
    ax.set_xlabel('t')
    ax.set_ylabel('$\phi_M(t)$')
    ax.set_title('Approximate DOS using KPM')
    ax.legend()

    return fig

def kernel_polynomial_method(A, t_points, M, nvec):
    # Schritt 1: Initialisierung von ζ_k
    zeta = np.zeros(M + 1)
    
    # Schritt 2: Hauptschleife über die Anzahl der Zufallsvektoren
    for L in range(nvec):
        # Schritt 3: Auswählen eines neuen Zufallsvektors
        v_0 = np.random.randn(A.shape[0])
        
        # Schritt 4: Schleife über k
        v_k_minus_1 = np.zeros_like(v_0)
        v_k = v_0
        for k in range(M + 1):
            zeta[k] += np.dot(v_0, v_k)
            
            # Schritt 6: Drei-Term Rekurrenz
            if k == 0:
                v_k_plus_1 = A @ v_k
            else:
                v_k_plus_1 = 2 * A @ v_k - v_k_minus_1
            
            v_k_minus_1 = v_k
            v_k = v_k_plus_1
    
    # Schritt 8: Normierung von ζ_k
    zeta /= nvec
    
    # Schritt 9: Jackson-Dämpfung
    mu_k = 2/(A.shape[0] * pi) * zeta
    mu_k[0] = 1/(A.shape[0] * pi) * zeta[0]
    
    # Schritt 10: Evaluierung der Dichte der Zustände (DOS)
    dos = np.zeros_like(t_points[1:-1])
    for i, t in enumerate(t_points[1:-1]):
        T_k = np.cos(np.arange(M + 1) * np.arccos(t))
        dos[i] = 1/sqrt(1 - t**2) * np.sum(mu_k * T_k)
    
    return dos

# Titel der App
st.title("Kernel-Polynom-Methode")

st.sidebar.header("Inputvariablen")

# Eingabe für die Dimension der Matrix A
st.session_state["n"] = st.sidebar.slider("Dimension der Matrix A (nxn):", min_value=1, max_value=100, value=4, step=1)
st.session_state["t_i"] = st.sidebar.slider("Anzahl der Auswertungen (t_i):", min_value=10, max_value=10000, value=100, step=10)
st.session_state["M"] = st.sidebar.slider("Grad des Erweiterungspolynoms (M):", min_value=5, max_value=100, value=5, step=5)
st.session_state["n_vec"] = st.sidebar.slider("Anzahl der zufällig generierten Vektoren (n_vec):", min_value=1, max_value=100, value=1,step=5)

if st.button("Generiere Zufallsmatrix"):

    # Generiere eine zufällige symmetrische Matrix
    A = generate_symmetric_matrix(st.session_state["n"])
    A = shift_eigenvalues(A)
    st.session_state["A"] = A
    st.session_state["eigenvalues"] = np.linalg.eigvalsh(A)

if "A" in st.session_state:
    with st.expander("Zufällige Matrix:", expanded=True):
        st.write(st.session_state["A"])
    with st.expander("Eigenwerte überprüfen"):
        st.write(st.session_state["eigenvalues"][None,:])

    # Schritt 1
    if st.button("# Schritt 1: Initalisierung der ζ_ks"):
        st.code("zeta = np.zeros(M + 1)")
        st.session_state["zeta"] = np.zeros(st.session_state["M"] + 1)
    if "zeta" in st.session_state:
        with st.expander("ζ_ks"):
            st.write(st.session_state["zeta"][None,:])

    # Schritt 2
    if "zeta" in st. session_state and st.button("# Schritt 2: Hauptschleife über die Anzahl der Zufallsvektoren"):
        st.code("for l in range(n_vec):\n\tv_0 = np.random.randn(A.shape[0])")
        st.session_state["l"] = 0
        st.session_state["v_0"] = np.random.randn(st.session_state["A"].shape[0])
    if "l" in st.session_state and "v_0" in st.session_state:
        with st.expander("l und v_0"):
            col1, col2 = st.columns([1, 3])
            with col1:
                st.write("l = ", st.session_state["l"])
            with col2:
                st.write(pd.DataFrame(st.session_state["v_0"], columns=["v_0"]))

    # Schritt 3
    if "v_0" in st.session_state and st.button("# Schritt 3: Schleife über k"):
        st.code("""v_k_minus_1 = np.zeros_like(v_0)
v_k = v_0
for k in range(M + 1):
    zeta[k] += np.dot(v_0, v_k)
    # Drei-Term Rekursion
    if k == 0:
        v_k_plus_1 = A @ v_k
    else:
        v_k_plus_1 = 2 * A @ v_k - v_k_minus_1
                
    v_k_minus_1 = v_k
    v_k = v_k_plus_1""")
        st.session_state["k"] = 0
        st.session_state["v_k_minus_1"] = np.zeros_like(st.session_state["v_0"])
        st.session_state["v_k"] = st.session_state["v_0"]
    if "v_k_minus_1" in st.session_state and "v_k" in st.session_state and "zeta" in st.session_state:
        with st.expander("v_k_minus_1, v_k und ζ_ks"):
            col1, col2, col3, col4 = st.columns([1, 2, 2, 2])
            with col1:
                st.write("k = ", st.session_state["k"])
            with col2:
                st.write(pd.DataFrame(st.session_state["v_k_minus_1"], columns=["v_k_minus_1"]))
            with col3:
                st.write(pd.DataFrame(st.session_state["v_k"], columns=["v_k"]))
            with col4:
                st.write(pd.DataFrame(st.session_state["zeta"], columns=["ζ_ks"]))
    if "k" in st.session_state and st.button("k + 1") and st.session_state["k"] < st.session_state["M"] + 1:
        sizes = [2, 1, 2, 1, 5, 1, 3, 1, 1, 4, 1, 4, 1, 1, 9]
        col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11, col12, col13, col14, col15 = st.columns(sizes, vertical_alignment='center')
        with col1:
            st.write(f"ζ_k")
        with col2:
            st.write("=")
        with col3:
            st.write(f"ζ_k")
        with col4:
            st.write("\+")
        with col5:
            st.write(f"⟨v_0, v_k⟩")
        with col6:
            st.write("=")
        with col7:
            st.write(st.session_state["zeta"][st.session_state["k"]])
        with col8:
            st.write("\+")
        with col9:
            st.write("⟨")
        with col10:
            st.write(st.session_state["v_0"])
        with col11:
            st.write(",")
        with col12:
            st.write(st.session_state["v_k"])
        with col13:
            st.write("⟩")
        with col14:
            st.write("=")
        st.session_state["zeta"][st.session_state["k"]] += np.dot(st.session_state["v_0"], st.session_state["v_k"])
        with col15:
            st.write(st.session_state["zeta"][st.session_state["k"]])

        if st.session_state["k"] == 0:
            v_k_plus_1 = st.session_state["A"] @ st.session_state["v_k"]
            col1, col2, col3, col4, col5, col6, col7 = st.columns([2, 1, 7, 1, 3, 1, 3], vertical_alignment='center')
            with col1:
                st.write("A @ v_0")
            with col2:
                st.write("=")
            with col3:
                st.write(st.session_state["A"])
            with col4:
                st.write("@")
            with col5:
                st.write(st.session_state["v_0"])
            with col6:
                st.write("=")
            with col7:
                st.write(v_k_plus_1)
        else:
            v_k_plus_1 = 2 * st.session_state["A"] @ st.session_state["v_k"] - st.session_state["v_k_minus_1"]
            st.write("2 * A @ v_k - v_k_minus_1")
            col1, col2, col3, col4, col5, col6, col7, col8, col9, col10 = st.columns([1, 1, 1, 10, 1, 3, 1, 3, 1, 3], vertical_alignment='center')
            with col1:
                st.write("=")
            with col2:
                st.write("2")
            with col3:
                st.write("\*")
            with col4:
                st.write(st.session_state["A"])
            with col5:
                st.write("@")
            with col6:
                st.write(st.session_state["v_k"])
            with col7:
                st.write("\-")
            with col8:
                st.write(st.session_state["v_k_minus_1"])
            with col9:
                st.write("=")
            with col10:
                st.write(v_k_plus_1)
        st.session_state["v_k_minus_1"] = st.session_state["v_k"]
        st.session_state["v_k"] = v_k_plus_1
        st.session_state["k"] += 1

    # Schritt 4
    if "k" in st.session_state and st.session_state["k"] == st.session_state["M"] + 1 and st.button("# Schritt 4: Normierung der ζ_k"):
        st.code("zeta /= n_vec")
        st.session_state["zeta"] /= st.session_state["n_vec"]
        if "zeta" in st.session_state:
            with st.expander("ζ_ks"):
                st.write(st.session_state["zeta"][None,:])
        st.session_state["zeta_normed"] = True
    
    # Schritt 5
    if "zeta_normed" in st.session_state and st.button("# Schritt 5: Berechnung der mu_k"):
        st.code("""mu = 2/(n * pi) * zeta
mu[0] = 2/(n * pi) * zeta[0]""")
        st.session_state["mu"] = 2/(st.session_state["n"] * pi) * st.session_state["zeta"]
        st.session_state["mu"][0] = 1/(st.session_state["n"] * pi) * st.session_state["zeta"][0]
    if "mu" in st.session_state:
        with st.expander("μ_ks"):
            st.write(st.session_state["mu"][None,:])

    # Schritt 6
    if "mu" in st.session_state and st.button("# Schritt 6: Auswertung der Spektraldichte"):
        st.code("""dos = np.zeros_like(t_points)
for i, t_i in enumerate(t_points):
T_k = np.cos(np.arange(M + 1) * np.arccos(t_i))
dos[i] = np.sum(mu_k * T_k)""")
        st.session_state["t_points"] = np.linspace(-1, 1, st.session_state["t_i"])[1:-1]
        st.session_state["dos"] = np.zeros_like(st.session_state["t_points"])
        for i, t_i in enumerate(st.session_state["t_points"]):
            st.session_state["T"] = np.cos(np.arange(st.session_state["M"] + 1) * np.arccos(t_i))
            st.session_state["dos"][i] = 1/sqrt(1 - t_i**2) * np.sum(st.session_state["mu"] * st.session_state["T"])
    if "dos" in st.session_state:
        with st.expander("Spektraldichte"):
            st.write(st.session_state["dos"][None,:])
    
    # Schritt 7
    if "dos" in st.session_state and st.button("# Schritt 7: Plotten der Spektraldichte"):
        st.code("""plt.plot(t_points, dos, label='DOS')
plt.xlabel('t')
plt.ylabel('$\phi_M(t)$')
plt.title('Approximate DOS using KPM')
plt.legend()
plt.show()""")
        # Erstellen der Matplotlib-Figur
        st.session_state["fig"] = generate_dos_plot(st.session_state["t_points"], st.session_state["dos"])
        # Anzeigen der Figur in Streamlit
        st.pyplot(st.session_state["fig"])
    
    if st.sidebar.button("Automatisch berechnen"):
        A = generate_symmetric_matrix(st.session_state["n"])
        A = shift_eigenvalues(A)
        st.session_state["A"] = A
        dos = kernel_polynomial_method(st.session_state["A"], st.session_state["t_points"], st.session_state["M"], st.session_state["n_vec"])
        st.session_state["dos"] = dos
        st.session_state["fig"] = generate_dos_plot(st.session_state["t_points"], st.session_state["dos"])
        st.pyplot(st.session_state["fig"])