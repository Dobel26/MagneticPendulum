import numpy as np
from scipy.fft import fft, ifft
from scipy.special import gamma, factorial


def remap(x, a, b, c, d): 
    """ Remaps x from interval [a, b] into interval [c, d]
    """
    return (x - a) / (b - a) * (d - c) + c

def fourier_diff_D(N):
    """ Generates the Fourier Differentiation Matrix D for N nodes, 
        works for both odd and even ;)
    """
    def val_kj(k, j): 
        if N % 2 == 0:  # N EVEN
            return 0.5 * (-1)**(k + j) / np.tan((k - j) * np.pi / N) if k != j else 0
        else:           # N ODD
            return 0.5 * (-1)**(k + j) / np.sin((k - j) * np.pi / N) if k != j else 0
    
    # Off diagonal
    D = np.array([[val_kj(k, j) for j in range(N)] for k in range(N)])
    # Negative sum trick for diagonal - stability reasons, Kopriva 55
    for k in range(N):
        # Better stability is obtained by summing smallest-in-magnitude terms first
        sorted_idx = np.argsort(np.abs(D[k, :]))
        sorted_row = D[k, :][sorted_idx]
        for j in range(N):
            D[k, k] -= sorted_row[j]
    
    return D

def fourier_diff(u, Omega, n):
    """ Calculates the n'th derivative of a function using the fourier 
        coefficients from FFT
    """
    u_hat = (1j * Omega)**n * fft(u)
    u_hat = ifft(u_hat)
    return u_hat

def jacobi_gq(alpha, beta, N):
    """ Compute the N'th order Gauss quadrature points x, and weights w associated 
        with the Jacobi polynomial of type (alpha, beta) > -1 ( <> -0.5 ).
        
        Translated from Allan's MATLAB implementation into Python
    """
    if (N == 0):
        x = np.array([-(alpha - beta) / (alpha + beta + 2)])
        w = np.array([2])
        return x, w
    
    ## Constructing symmetric matrix from recurrence
    # Diagonal
    h1 = 2 * np.arange(N+1) + alpha + beta
    diag = np.diag(-(alpha**2 - beta**2) / (h1 + 2) / h1)
    # Off diagnoal
    n = np.arange(1, N+1)
    off_diag = np.diag(2 / (h1[:-1] + 2) * np.sqrt(
        n * (n + alpha + beta) * (n + alpha) * (n + beta) / (h1[:-1] + 1) / (h1[:-1] + 3)
    ), 1)
    # Assembling matrix
    J = diag + off_diag + off_diag.T
    if (alpha + beta < 10 * np.finfo(float).eps):
        J[0, 0] = 0.0
    
    # Compute quadrature by eigenvalue solve
    eigen_values, eigen_vectors = np.linalg.eigh(J)
    x = eigen_values
    
    w = (eigen_vectors[0,:]**2 * 2**(alpha + beta + 1) 
         / (alpha + beta + 1) * gamma(alpha + 1) * gamma(beta + 1) 
         / gamma(alpha + beta + 1)
    )
    
    return x, w

def jacobi_gl(alpha, beta, N):
    """ Compute the N'th order Gauss Lobatto quadrature points x associated with the
        Jacobi polynomial of type (alpha, beta) > -1 ( <> -0.5)
        
        **Dependent on:** `jacobi_gq` 
        
        Translated from Allan's MATLAB implementation into Python
    """    
    x = np.zeros(N+1)
    if (N==1):
        x[0] = -1
        x[1] = 1
        return x
    
    xint, _ = jacobi_gq(alpha + 1, beta + 1, N - 2)
    # x = [-1, xint.T, 1].T
    x = np.ones(len(xint) + 2)
    x[0] = -1
    x[1:-1] = xint
    
    return x

def lobatto_weights(abscissas):
    """ Compute the weights for quadrature nodes using Gauss-Lobatto quadrature with a given abscissas
    
        **Dependent on:** `legendre_pol`
    """
    N = len(abscissas) - 1
    LN = legendre_pol(abscissas, N)[-1]
    return 2 / (N * (N + 1) * LN**2)

def jacobi_pol(x, alpha, beta, N):
    """ Compute the first N+1 Jacobi polynomials of order atmost N of type (alpha, beta) > -1 ( <> -0.5)
    """
    p0 = np.ones(len(x))
    p1 = 0.5 * (alpha - beta + (alpha + beta + 2) * x)
    p = [p0, p1]
    
    if (N == 0):
        return np.array([p0])
    
    if (N == 1):
        return np.array(p)
    
    for i in range(N-1):
        n = i + 1
        d = 2 * n + alpha + beta
        
        anm1 = 2 * (n + alpha) * (n + beta) / ((d + 1) * d)
        an = (alpha**2 - beta**2) / ((d + 2) * d)
        anp1 = 2 * (n + 1) * (d - n + 1) / ((d + 2) * (d + 1))
        
        pnp1 = (an + x) * p[-1] - anm1 * p[-2]
        
        p.append(pnp1 / anp1)
    
    return np.array(p)

def chebyshev_pol(x, N):
    """ Compute the first Chebyshev polynomials of order atmost N.
        
        **Dependent on:** `jacobi_pol`
    """
    pols = jacobi_pol(x, -0.5, -0.5, N)
    
    for n, pol in enumerate(pols):
        g_n = gamma(n + 1) * gamma(0.5) / gamma(n + 0.5)
        pol *= g_n
    
    return pols

def grad_jacobi_pol(x, alpha, beta, n):
    """ Computes the first derivative of the n'th Jacobi Polynomial
    
        **Dependent on:** `jacobi_pol`
    """
    return 0.5 * (alpha + beta + n + 1) * jacobi_pol(x, alpha + 1, beta + 1, n - 1)[-1]

def legendre_pol(x, N):
    """ Compute the first N+1 Legendre polynomials of order atmost N.

        **Dependent on:** `jacobi_pol` 
    """
    return jacobi_pol(x, 0, 0, N)

def grad_legendre_pol(x, n):
    """ Computes the first derivative of the n'th Jacobi Polynomial
    
        **Dependent on:** `jacobi_pol`
    """
    return grad_jacobi_pol(x, 0, 0, n)

def jacobi_pol_orthonormal(x, alpha, beta, N):
    """ Compute the first N+1 orthonormalized Jacobi polynomials of order atmost N of type (alpha, beta) > -1 ( <> -0.5)
    
        **Dependent on:** `jacobi_pol`
    """
    pols = jacobi_pol(x, alpha, beta, N)
    for n, pol in enumerate(pols):
        if alpha == 0 and beta == 0:
            gn = 2 / (2 * n + 1)
        elif alpha == 1 and beta == 1:
            gn = 8 * (n + 1) / (2 * n + 3) / (n + 2)
        else:
            gn = 2**(alpha + beta + 1) * gamma(n + alpha + 1) * gamma(n + beta + 1)
            gn /= factorial(n) * (2 * n + alpha + beta + 1) * gamma(n + alpha + beta + 1)
            
        pol /= np.sqrt(gn)
        
    return pols
    
def legendre_pol_orthonormal(x, N):
    """ Compute the first N+1 orthonormalized Legendre polynomials of order atmost N.
    
        **Dependent on:** `jacobi_pol_orthonormal`
    """
    return jacobi_pol_orthonormal(x, 0, 0, N)

def grad_legendre_pol_orthonormal(x, n):
    """ Compute the derivative of the n'th orthonormal legendre polynomial.
    
        **Dependent on:** `jacobi_pol_orthonormal`
    """
    return np.sqrt(n * (n + 1)) * jacobi_pol_orthonormal(x, 1, 1, n-1)[-1]

def pol_coef(abscissas, weights, func, alpha, beta, orthonormal: bool=False):
    """ Compute the modal expansion coefficients related to the orthonormal Jacobi polynomials, evaluated in 
        a given abscissas. **Returns** `(coefficients, basis polynomials)`
    
        **Dependent on:** `jacobi_pol`, `jacobi_pol_orthonormal`
    """
    N = len(abscissas)
    coef = np.zeros(N)
    f = func(abscissas)
    if orthonormal:
        phi = jacobi_pol_orthonormal(abscissas, alpha, beta, N - 1)
    else:
        phi = jacobi_pol(abscissas, alpha, beta, N - 1)
    
    for n in range(N):
        for j in range(N):
            coef[n] += f[j] * phi[n, j] * weights[j]
        
        if not orthonormal:
            if alpha == 0 and beta == 0:  # Numerical stability for Legendre Polynomials
                gn = 2 / (2 * n + 1)
            else:
                gn = 2**(alpha + beta + 1) * gamma(n + alpha + 1) * gamma(n + beta + 1)
                gn /= factorial(n) * (2 * n + alpha + beta + 1) * gamma(n + alpha + beta + 1)
        
            coef[n] /= gn
    
    return coef, phi

def vandermonde(abscissas):
    """ Compute the Vandermonde matrix V based on orthonormal Legendre polynomials from given abscissas.
    
        **Dependent on:** `legendre_pol_orthonormal`
    """
    N = len(abscissas)
    return legendre_pol_orthonormal(abscissas, N-1).T

def vandermonde_x(abscissas):
    """ Compute the Vandermonde differentiation matrix Vx based on differentiated orthonormal Legendre 
        polynomials from given abscissas.
    
        **Dependent on:** `grad_legendre_pol_orthonormal`
    """
    N = len(abscissas)
    return np.array([grad_legendre_pol_orthonormal(abscissas, n) for n in range(N)]).T

def euler(u, rhs, dt):
    return u + rhs(u) * dt

def rk4(u, t, rhs, dt):
    k1 = rhs(u)
    k2 = rhs(u + dt * k1 / 2)
    k3 = rhs(u + dt * k2 / 2)
    k4 = rhs(u + dt * k3)
    
    return u + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
