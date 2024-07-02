import streamlit as st
import numpy as np

# Titel der App
st.title("Erwartungswert einer Zufallsmatrix")

st.sidebar.header("Beispiel mit einem Vektor")

# Eingabe für die Dimension des Vektors
n = st.sidebar.number_input("Dimension des Vektors (n):", min_value=1, max_value=10, value=3, step=1)

if st.sidebar.button("Generiere Zufallsvektor"):
    
    # Generiere einen zufälligen Vektoren
    v = np.random.randn(n)[:, None]

    # Berechne den Erwartungswert des Vektors (Mittelwert über alle Einträge)
    expected_v = np.mean(v)

    # Berechne das äußere Produkt für den Vektor
    outer_product = np.array([np.outer(v, v)])

    # Berechne den Erwartungswert des äußeren Produkts (Mittelwert über alle Stichproben)
    expected_value = np.mean(v)

    # Ausgabe des Ergebnisses
    st.write("Zufälliger Vektor:")
    st.write(v)
    st.write("Erwartungswert des Vektors:")
    st.write(expected_v)
    st.write("Das Produkt M = v * v^T")
    st.write(np.array(np.outer(v, v)))
    st.write("Der Erwartungswert der Matrix M ist für eine Stichprobe genau M")

st.sidebar.header("Sampling über viele Stichproben")

# Eingabe für Anzahl der Stichproben
num_samples = st.sidebar.slider("Anzahl der Stichproben:", min_value=10, max_value=1000, value=350, step=10)

# Button zur Berechnung
if st.sidebar.button("Berechne Erwartungswert"):

    # Generiere num_samples zufällige Vektoren
    random_vectors = np.random.randn(num_samples, n)

    # Berechne den Erwartungswert der Vektoren (Mittelwert über alle Stichproben)
    expected_vs = np.mean(random_vectors)

    # Berechne das äußere Produkt für jeden Vektor
    outer_products = np.array([np.outer(v, v) for v in random_vectors])

    # Berechne den Erwartungswert der äußeren Produkte (Mittelwert über alle Stichproben)
    expected_value = np.mean(outer_products, axis=0)

    # Ausgabe des Ergebnisses
    st.write("Erwartungswert der Vektoren:")
    st.write(expected_vs)

    st.write("Erwartungswert der Produkte M = v*v^t:")
    st.write(expected_value)