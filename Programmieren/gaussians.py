import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# Gaussian function as defined in equation 2.3
def gaussian(t, sigma):
    return (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-t**2 / (2 * sigma**2))

# Generate a random symmetric matrix and compute its eigenvalues
def generate_eigenvalues(matrix_size):
    A = np.random.randn(matrix_size, matrix_size)
    A = (A + A.T) / 2  # Make the matrix symmetric
    eigenvalues = np.linalg.eigvalsh(A)
    return np.sort(eigenvalues)

# Generate DOS plots
def generate_dos_plots(eigenvalues, sigmas):
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()
    
    j = np.arange(1, len(eigenvalues) + 1)
    axes[0].plot(j, eigenvalues, 'o-', markersize=2)
    axes[0].set_title('Eigenvalues')
    axes[0].set_xlabel('j')
    axes[0].set_ylabel('$\lambda_j$')
    
    for ax, sigma in zip(axes[1:], sigmas):
        blurred_eigenvalues = gaussian_filter1d(eigenvalues, sigma)
        ax.plot(blurred_eigenvalues)
        ax.set_title(f'κ = {sigma:.2f}, σ = {sigma:.2f}')
        ax.set_xlabel('t')
        ax.set_ylabel('$\phi(t)$')
    
    plt.tight_layout()
    return fig

# Streamlit app
#def main():
st.title("Eigenvalues and Regularized DOS")

# Matrix size input
matrix_size = st.sidebar.slider('Matrix Size', min_value=100, max_value=1000, value=700, step=50)
    
# Sigma values input
sigma1 = st.sidebar.slider('Sigma 1', min_value=0.1, max_value=2.0, value=0.35, step=0.01)
sigma2 = st.sidebar.slider('Sigma 2', min_value=0.1, max_value=2.0, value=0.52, step=0.01)
sigma3 = st.sidebar.slider('Sigma 3', min_value=0.1, max_value=2.0, value=0.71, step=0.01)
sigma4 = st.sidebar.slider('Sigma 4', min_value=0.1, max_value=2.0, value=0.96, step=0.01)

sigmas = [sigma1, sigma2, sigma3, sigma4]

eigenvalues = generate_eigenvalues(matrix_size)

fig = generate_dos_plots(eigenvalues, sigmas)
st.pyplot(fig)

# if __name__ == "__main__":
#     main()