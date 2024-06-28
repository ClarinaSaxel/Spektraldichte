import numpy as np

def generate_symmetric_matrix(n):
    A = np.random.randn(n, n)
    #print(f'A = {A}\nA.T = {A.T}\nA + A.T = {A + A.T}\n(A + A.T)/2 = {(A + A.T)/2}\n')
    A=(A + A.T)/2
    return A

def shift_eigenvalues(A):
    eigenvalues, _ = np.linalg.eigh(A)
    c = (max(eigenvalues) + min(eigenvalues))/2
    d = (max(eigenvalues) - min(eigenvalues))/2
    B = (A - c * np.identity(len(A)))/d
    return B

if __name__ == '__main__':

    A = generate_symmetric_matrix(100)
    B = shift_eigenvalues(A)   
    beigenvalues, beigenvectors = np.linalg.eigh(B)
    print(f'B = {B}\nbeigenvalues = {beigenvalues}')