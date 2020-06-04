from lib import *
import numpy as np
import cv2

def Encrypt(A):
    # Step 1
    (m, n) = A.shape
    s = Entropy(A.flatten())

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
            W[i][j] = (m*n+i+j) % 256
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
            C[:, i] = (R[:, i]+d*K[:, i]+K[:, d]) % 256
        else:
            C[:, i] = (R[:, i]+d*C[:, i-1]+d*K[:, i]+K[:, d]) % 256
    return C
if __name__ == '__main__':
    A = cv2.imread('./img/test_lena_256.bmp', cv2.IMREAD_GRAYSCALE)
    # 2 rounds
    C = Encrypt(A)
    C = Encrypt(C)
    cv2.imwrite('./img/cipher.bmp', C)
