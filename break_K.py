import cv2
import numpy as np
from scipy.stats import entropy

from lib import *
from encrypt import Encrypt

np.random.seed(7122)


def GenRandomImage(origin_img):
    (M, N) = origin_img.shape
    random_img = np.zeros(M*N, dtype=origin_img.dtype)
    cnt = np.zeros(256, dtype=int)
    for x in origin_img.flatten():
        cnt[x] += 1
    np.random.shuffle(cnt)
    p = 0
    for x in range(256):
        for c in range(cnt[x]):
            random_img[p] = x
            p += 1
    np.random.shuffle(random_img)
    random_img = random_img.reshape(M, N)
    return random_img

def GetR(A):
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
    return R

def GetK(A):
    (m, n) = A.shape
    xp_0, yp_0 = UpdateKey2(x0, y0, xp0, yp0)
    K_seq = LASM2D(mu, xp_0, yp_0, m*n)
    K = K_seq.reshape(A.shape)
    K = np.ceil(K*(10**14)) % 256
    K = K.astype('uint8')
    return K

Q = np.zeros((256, 256), dtype=int)
for i in range(1, 256):
    for j in range(1, 256):
        for q in range(1, 256):
            if (i + q * 256) % j == 0:
                Q[i, j] = ((i + q * 256) // j) % 256

def Gaussian(A, B, x, p_start, mod=256):
    N, M = A.shape
    assert(N == M)
    ### Gaussian Elimination
    for j in range(M):
        ### Find denominator
        column = A[j:, j]
        min_value = 256
        min_position = -1
        while True:
            candidates = np.where(column % 2 == 1)[0]
            if len(candidates) != 0:
                for c in candidates:
                    if A[j+c, j] < min_value:
                        min_value = A[j+c, j]
                        min_position = j+c
                break
            column = column // 2
        ### All non-zero numbers are power of 2
        if min_position == -1:
            for i in range(j, N):
                if A[i, j] == 0:
                    continue
                if A[i, j] < min_value:
                    min_value = A[i, j]
                    min_position = i
        ### Move the denominator row to the top
        A[[j, min_position]] = A[[min_position, j]]
        B[[j, min_position]] = B[[min_position, j]]
        ### Substract each row below with the denominator row
        for i in range(j+1, N):
            if A[i, j] != 0:
                m = Q[A[i, j], A[j, j]]
                A[i, :] = (A[i, :] - A[j, :] * m) % 256
                B[i] = (B[i] - B[j] * m) % 256

    # for j in reversed(range(M)):
    #     for i in range(j):
    #         if A[i, j] != 0:
    #             m = Q[A[i, j], A[j, j]]
    #             if m == 0:
    #                 print('NO!!!!!!!!!')
    #             A[i, :] = (A[i, :] - A[j, :] * m) % 256
    #             B[i] = (B[i] - B[j] * m) % 256

    # x = np.zeros(M, dtype=int)
    for j in reversed(range(p_start+1)):
        b = (B[j] - np.dot(A[j,:], x)) % 256
        a = A[j,j]
        possible_solution = []
        for i in range(256):
            if (a * i) % 256 == b:
                possible_solution.append(i)
        if len(possible_solution) == 1:
            x[j] = possible_solution[0]
        else:
            ### return and do next round
            return A, B, x, j, possible_solution

    return A, B, x, -1, []

def RetrieveEquations(C, R):
    (M, N) = C.shape
    DI = np.zeros((N, N), dtype='uint8')
    D = np.zeros(N, dtype='uint8')
    for i in reversed(range(N)):
        if i < N-1:
            d = np.ceil(Entropy(R[:, i+1:])*(10**14)) % N
            d = d.astype(int)
        else:
            d = 0
        DI[i, i] = d + 1
        DI[i, d] = 1
        if i == 0:
            D[i] = (C[0, i] - R[0, i]) % 256
        else:
            D[i] = C[0, i] - R[0, i] - (d + 1) * C[0, i-1]
    return DI, D




origin_img = cv2.imread('img/test_lena_256.bmp', cv2.IMREAD_GRAYSCALE)
origin_img = cv2.resize(origin_img, (16, 16))
# cv2.imwrite('8x8.bmp', origin_img)
(M, N) = origin_img.shape

_K = np.zeros(N, dtype=int)
p_start = N-1
while True:
    random_img = GenRandomImage(origin_img)
    # print('{:.40f}'.format(Entropy(origin_img)))
    # print('{:.40f}'.format(Entropy(random_img)))

    R_random = GetR(random_img)
    C_random = Encrypt(random_img)
    # print(C_random)

    DI, D = RetrieveEquations(C_random, R_random)
    # print(DI)
    # print(D)

    _A, _B, _K, p_stop, possible_solution = Gaussian(DI, D, _K, p_start)
    # print(_A)
    # print(_B)
    print(_K)
    print(p_stop)
    print(possible_solution)
    # print(np.dot(_A, _K) % 256)
    # print(np.dot(_A, _K) % 256 == _B)

    if p_stop == -1:
        break
    else:
        p_start = p_stop

print('final:')
print(_K)

K = GetK(random_img)
print(K)
# print(np.dot(_A, K[0]) % 256 == _B)
