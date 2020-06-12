from lib import *
import numpy as np
import cv2
import sys
import concurrent.futures

def Work(start, end, R, n):
    d_list = []
    for i in range(start, end):
        d = np.ceil(Entropy(R[i+1:])*(10**14)) % n
        d_list.append(d.astype(int))
    return d_list

def FastEncrypt(A, n_pool=8):
    # Step 1
    (m, n) = A.shape
    s = Entropy(A)

    x_0, y_0 = UpdateKey1(x0, y0, xp0, yp0, s)
    P_seq = LASM2D(mu, x_0, y_0, m*n)
    P = P_seq.reshape(A.shape)

    # Step 2
    a = np.ceil((x0+y0+1)*(10**7)) % (m)
    b = np.ceil((xp0+yp0+2)*(10**7)) % (n)
    u = P[int(a), :]
    v = P[:, int(b)]
    up = np.ceil(u*(10**14)) % (n)
    vp = np.ceil(v*(10**14)) % (m)
    up = up.astype(int)
    vp = vp.astype(int)
    Uniq(up)
    Uniq(vp)
    upt = np.empty_like(up)
    vpt = np.empty_like(vp)
    for i in range(n):
        upt[up[i]] = i
    for i in range(m):
        vpt[vp[i]] = i
    B = np.transpose(np.transpose(A)[upt]).copy()
    B = B[vpt]

    # Step 3
    W = np.array([[(m*n+(i+1)+(j+1)) for j in range(n)] for i in range(m)], dtype='uint8')
    R = B+W

    # Step 4
    xp_0, yp_0 = UpdateKey2(x0, y0, xp0, yp0)
    K_seq = LASM2D(mu, xp_0, yp_0, m*n)
    K = K_seq.reshape(A.shape)
    K = np.ceil(K*(10**14)) % 256
    K = K.astype('uint8')
     
    # Step 5
    C = np.zeros((n,m), dtype='uint8')
    R = np.transpose(R)
    K = np.transpose(K)
    d = []
    workers = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        gap = n // n_pool
        for i in range(0, n-1, gap):
            if i + gap >= n-1:
                workers.append(executor.submit(Work, i, n-1, R, n))
            else:
                workers.append(executor.submit(Work, i, i+gap, R, n))
        executor.shutdown(wait=True)
    for w in workers:
        d.extend(w.result())
    # index = n-1
    d.append(0)
        
    for i in range(n):
        if i == 0:
            C[i] = (R[i]+(d[i]+1)*K[i]+K[d[i]])
        else:
            C[i] = (R[i]+(d[i]+1)*C[i-1]+(d[i]+1)*K[i]+K[d[i]])

    return np.transpose(C)
def Encrypt(A):
    # Step 1
    (m, n) = A.shape
    s = Entropy(A)

    x_0, y_0 = UpdateKey1(x0, y0, xp0, yp0, s)
    P_seq = LASM2D(mu, x_0, y_0, m*n)
    P = P_seq.reshape(A.shape)

    # Step 2
    a = np.ceil((x0+y0+1)*(10**7)) % (m)
    b = np.ceil((xp0+yp0+2)*(10**7)) % (n)
    u = P[int(a), :]
    v = P[:, int(b)]
    up = np.ceil(u*(10**14)) % (n)
    vp = np.ceil(v*(10**14)) % (m)
    up = up.astype(int)
    vp = vp.astype(int)
    Uniq(up)
    Uniq(vp)
    B = np.zeros(A.shape, dtype='uint8')
    tmp = np.zeros(A.shape, dtype='uint8')
    for i in range(n):
        tmp[:, up[i]] = A[:, i]
    for i in range(m):
        B[vp[i], :] = tmp[i, :]
    
    # Step 3
    W = np.zeros(A.shape, dtype='uint8')
    for i in range(m):
        for j in range(n):
            W[i][j] = (m*n+(i+1)+(j+1)) % 256
    R = (B+W) % 256

    # Step 4
    xp_0, yp_0 = UpdateKey2(x0, y0, xp0, yp0)
    K_seq = LASM2D(mu, xp_0, yp_0, m*n)
    K = K_seq.reshape(A.shape)
    K = np.ceil(K*(10**14)) % 256
    K = K.astype('uint8')

    # Step 5
    C = np.zeros(A.shape, dtype='uint8')
    for i in range(n):
        if i < n-1:
            d = np.ceil(Entropy(R[:, i+1:])*(10**14)) % n
            d = d.astype(int)
        else:
            d = 0
        if i == 0:
            C[:, i] = (R[:, i]+(d+1)*K[:, i]+K[:, d]) % 256
        else:
            C[:, i] = (R[:, i]+(d+1)*C[:, i-1]+(d+1)*K[:, i]+K[:, d]) % 256
    return C
if __name__ == '__main__':
    if len(sys.argv) >= 2:
        filename = sys.argv[1]
    else:
        filename = './img/test_lena_256.bmp'
    A = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    # 2 rounds
    C = FastEncrypt(A)
    C = FastEncrypt(C)
    cv2.imwrite('./img/cipher.bmp', C)
