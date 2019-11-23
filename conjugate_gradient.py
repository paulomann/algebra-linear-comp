import numpy as np

def conjugate_gradient(A,b,x0,N,TOL, C: np.ndarray = None):
    assert A.shape[0] == A.shape[1], "Matrix is not squared."
    C = C if C is not None else np.eye(A.shape[0])
    r0 = b - np.dot(A,x0)
    w0 = np.dot(C, r0)
    v0 = np.dot(C.T, w0)
    alpha = np.dot(r0, r0)
    k=1
    while(k<N):
        u = np.dot(A, v0)
        t = alpha / np.dot(u, v0)
        x1 = x0 + t * v0
        r1 = r0 - t * u
        w1 = np.dot(C, r1)
        if(np.linalg.norm(w1) <= TOL):
            break
        beta = np.dot(w1, w1)
        s = beta / alpha
        v1 = np.dot(C, w1) + s*v0
        alpha = beta
        x0, r0, v0 = x1, r1, v1
        k+=1
    return x1, k


if "__main__" == __name__:
    A = np.array([[1.,-1.,0.],\
                [-1.,2.,1.],\
                [0.,1.,5.]])
    b = np.array([3.,-3.,4.])
    guess = np.array([0.,0.,0.])
    x,n = conjugate_gradient(A,b,guess,100,0.00000001)
    print(f"GraCon = {x.round(2)} em {n} iterações.")