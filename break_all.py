import cv2
import numpy as np
np.random.seed(7125)

from lib import Entropy
from encrypt import Encrypt, FastEncrypt
from break_permutation import break_permutation
from break_K import break_K, GenerateRandomImage
from uv import up, vp


def MaliciousDecrypt(C, up, vp, K):
    # Step 1
    (m, n) = C.shape
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

    # Step 5
    A = np.zeros(C.shape, dtype='uint8')
    tmp = np.zeros(C.shape, dtype='uint8')
    for i in range(m):
        tmp[i, :] = B[vp[i], :]
    for i in range(n):
        A[:, i] = tmp[:, up[i]]

    return A

def break_all(C, A):
    up, vp = break_permutation(A)
    print(up, vp)
    # exit()
    # global up, vp
    # up = np.array(up)
    # vp = np.array(vp)
    K = break_K(A, up, vp)
    P = MaliciousDecrypt(C, up, vp, K)
    return P


if __name__ == '__main__':
    I = cv2.imread('img/couple.bmp', cv2.IMREAD_GRAYSCALE)
    I = cv2.resize(I, (260, 260))
    C = FastEncrypt(I)
    A = GenerateRandomImage(I)

    P = break_all(C, A)
    
    n_correct_pixel = np.sum(P == I)
    n_total_pixel = I.shape[0] * I.shape[1]
    if n_correct_pixel == n_total_pixel:
        print('Success!')
    else:
        print('Jizz...')
