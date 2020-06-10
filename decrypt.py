from lib import *
import numpy as np
import cv2

def Decrypt(C):
    # Step 1
    (m, n) = C.shape
    xp_0, yp_0 = UpdateKey2(x0, y0, xp0, yp0)
    K_seq = LASM2D(mu, xp_0, yp_0, m*n)
    K = K_seq.reshape(C.shape)
    K = np.ceil(K*(10**14)) % 256
    K = K.astype('uint8')
    
    # Step 2
    R = np.zeros(C.shape, dtype='uint8')
    for i in reversed(range(n)):
        if i < n-1:
            d = np.ceil(Entropy(R[:, i+1:])*(10**14)) % n
            d = d.astype(int)
        else:
            d = 0
        if i == 0:
            R[:, i] = (C[:, i]-(d+1)*K[:, i]-K[:, d]) % 256
        else:
            R[:, i] = (C[:, i]-(d+1)*C[:, i-1]-(d+1)*K[:, i]-K[:, d]) % 256

    # Step 3
    W = np.zeros(C.shape, dtype='uint8')
    for i in range(m):
        for j in range(n):
            W[i][j] = (m*n+(i+1)+(j+1)) % 256
    B = (R-W) % 256

    # Step 4
    
    s = Entropy(B.flatten())
    x_0, y_0 = UpdateKey1(x0, y0, xp0, yp0, s)
    P_seq = LASM2D(mu, x_0, y_0, m*n)
    P = P_seq.reshape(C.shape)

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

    # Step 5
    A = np.zeros(C.shape, dtype='uint8')
    tmp = np.zeros(C.shape, dtype='uint8')
    for i in range(m):
        tmp[i, :] = B[vp[i], :]
    for i in range(n):
        A[:, i] = tmp[:, up[i]]

    return A

if __name__ == '__main__':
    C = cv2.imread('./img/cipher.bmp', cv2.IMREAD_GRAYSCALE)
    # 2 rounds
    A = Decrypt(C)
    A = Decrypt(A)
    cv2.imwrite('./img/decrypt.bmp', A)
