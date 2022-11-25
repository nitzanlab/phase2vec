import torch
import numpy as np
from scipy.special import binom

def library_size(dim, poly_order, include_sine=False, include_exp=False, include_constant=True):
    """
    Calculate library size for a given polynomial order. Taken from `https://github.com/kpchamp/SindyAutoencoders`

    """
    l = 0
    for k in range(poly_order+1):
        l += int(binom(dim+k-1,k))
    if include_sine:
        l += dim
    if include_exp:
        l+= dim
    if not include_constant:
        l -= 1
    return l

def sindy_library(X, poly_order, include_sine=False, include_exp=False):
    """
    Generate a library of polynomials of order poly_order. Taken from `https://github.com/kpchamp/SindyAutoencoders`
    """
    X = torch.Tensor(X) if isinstance(X, (np.ndarray, np.generic)) else X
    m,n = X.shape # samples x dim
    l = library_size(n, poly_order, include_sine=include_sine, include_exp=include_exp, include_constant=True)
    library = torch.ones((m,l)).to(X.dtype)
    index = 1
    
    library_terms = []

    for i in range(n):
        library[:,index] = X[:,i]
        index += 1
        library_terms.append(r'$x_{}$'.format(i))

    if poly_order > 1:
        for i in range(n):
            for j in range(i,n):
                library[:,index] = X[:,i]*X[:,j]
                index += 1
                library_terms.append(r'$x_{}x_{}$'.format(i,j))


    if poly_order > 2:
        for i in range(n):
            for j in range(i,n):
                for k in range(j,n):
                    library[:,index] = X[:,i]*X[:,j]*X[:,k]
                    index += 1
                    library_terms.append(r'$x_{}x_{}x_{}$'.format(i,j,k))

    if poly_order > 3:
        for i in range(n):
            for j in range(i,n):
                for k in range(j,n):
                    for q in range(k,n):
                        library[:,index] = X[:,i]*X[:,j]*X[:,k]*X[:,q]
                        index += 1
                        library_terms.append(r'$x_{}x_{}x_{}x_{}$'.format(i,j,k,q))

                    
    if poly_order > 4:
        for i in range(n):
            for j in range(i,n):
                for k in range(j,n):
                    for q in range(k,n):
                        for r in range(q,n):
                            library[:,index] = X[:,i]*X[:,j]*X[:,k]*X[:,q]*X[:,r]
                            index += 1
                            library_terms.append(r'$x_{}x_{}x_{}x_{}x_{}$'.format(i,j,k,q,r))


    if include_sine:
        for i in range(n):
            library[:,index] = torch.sin(X[:,i])
            index += 1
            library_terms.append(r'$sin(x_{})$'.format(i))

    if include_exp: 
        for i in range(n):
            library[:,index] = torch.exp(-1*X[:,i])
            index += 1
            library_terms.append(r'$exp(-x_{})$'.format(i))


    # Consolidate exponents
    for t, term in enumerate(library_terms):
        if 'sin' in term or 'e' in term: continue
        all_exps = []
        for i in range(n):
            all_exps.append(term.count(str(i)))
        new_term = ''
        for e, exp in enumerate(all_exps):
            if exp == 0: continue
            if exp == 1:
                new_term += 'x_{}'.format(e)
            else:
                new_term += 'x_{}^{}'.format(e,exp)
        new_term = r'${}$'.format(new_term)
        library_terms[t] = new_term
    # Add constant term
    library_terms.insert(0,'1')
    return library, library_terms
