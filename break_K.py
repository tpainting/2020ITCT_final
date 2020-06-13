import cv2
import numpy as np

from lib import *
from encrypt import Encrypt, FastEncrypt


def FindSmallNumber(seq):
    if np.sum(seq) == 0:
        print('ALL ZERO!!!')
        return -1
    while True:
        mask = seq % 2
        if np.sum(mask) != 0:
            min_value = np.min(seq[mask.nonzero()])
            return np.where(seq == min_value)[0][0]
        seq = seq // 2

def ConstructQ():
    Q = np.zeros((256, 256), dtype=int)
    for i in range(1, 256):
        for j in range(1, 256):
            if i == j:
                Q[i,j] = 1
            if FindSmallNumber(np.array([i, j])) == 1 or j % 2 == 1:
                for q in range(256):
                    if i == (q * j) % 256:
                        Q[i,j] = q
    return Q

Q = ConstructQ()
print('ConstructQ finished.')

### Solve x from Ax = B
def GaussianEliminationSolve(A, B, X, p_start):
    A_M, A_N = A.shape
    assert(A_M == A_N)
    B_M = B.shape[0]
    assert(A_M == B_M)
    N = A_N

    ### Gaussian elimination
    for i in range(p_start+1):
        ### Find denominator
        pos = i + FindSmallNumber(A[i:,i])
        # print('FindSmallNumber:', pos)
        if pos == i - 1:
            # print('Gaussian elimination failed due to all-zero column')
            # return A, B, X, p_start
            print('Gaussian elimination encounter all-zero column, continue ...')
            continue
        ### Move the denominator row to row i
        A[[i, pos]] = A[[pos, i]]
        B[[i, pos]] = B[[pos, i]]
        ### Substract each row below with the denominator row
        for j in range(i+1, N):
            if A[j,i] != 0:
                # m = -1
                # for q in range(256):
                #     if A[i,i] * q % 256 == A[j,i]:
                #         m = q
                #         break
                # if m == -1:
                #     print('Can\'t find multiplier', A[i,i], A[j,i])
                #     exit()
                m = Q[A[j,i],A[i,i]]
                if m == 0:
                    print('multiplier equals to 0!', A[j,i], A[i,i])
                    exit()
                A[j,:] = (A[j,:] - A[i,:] * m) % 256
                B[j,:] = (B[j,:] - B[i,:] * m) % 256

    ### Solve upper triangular matrix
    for i in reversed(range(p_start+1)):
        a = A[i,i]
        b = (B[i,:] - np.dot(A[i,:], X)) % 256
        ### if a is even, then there are multiple solutions
        if a % 2 == 0:
            return A, B, X, i
        ### solve x in ax = b
        # for x in range(256):
            # if (a * x) % 256 == b:
                # X[i] = x
                # break
        X[i,:] = Q[b,a]
    
    return A, B, X, -1

### Retrieve equation system from R and C
def RetrieveEquations(R, C):
    R_M, R_N = R.shape
    C_M, C_N = C.shape
    assert(R_M == C_M and R_N == C_N)
    N = R_N
    M = R_M

    A = np.zeros((N, N), dtype=int)
    B = np.zeros((N, M), dtype=int)
    for i in reversed(range(N)):
        ### Retrieve d
        if i == N - 1:
            d = 0
        else:
            d = np.ceil(Entropy(R[:,i+1:]) * (10 ** 14)) % N
            d = d.astype(int)
        A[i,i] += d + 1
        A[i,d] += 1
        ### Retrieve D
        if i == 0:
            B[i,:] = (C[:,i] - R[:,i]) % 256
        else:
            B[i,:] = (C[:,i] - R[:,i] - (d + 1) * C[:,i-1]) % 256

    A, B = A % 256, B % 256

    return A, B

def GetR(I):
    # Step 1
    (m, n) = I.shape
    s = Entropy(I)

    x_0, y_0 = UpdateKey1(x0, y0, xp0, yp0, s)
    P_seq = LASM2D(mu, x_0, y_0, m*n)
    P = P_seq.reshape(I.shape)

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
    B = np.zeros(I.shape, dtype='uint8')
    tmp = np.zeros(I.shape, dtype='uint8')
    for i in range(n):
        tmp[:, up[i]] = I[:, i]
    for i in range(m):
        B[vp[i], :] = tmp[i, :]
    
    # Step 3
    W = np.zeros(I.shape, dtype='uint8')
    for i in range(m):
        for j in range(n):
            W[i][j] = (m*n+(i+1)+(j+1)) % 256
    R = (B+W) % 256
    
    return R

def GetRFromPermutation(I, up, vp):
    # Step 1
    (m, n) = I.shape
    s = Entropy(I)

    # x_0, y_0 = UpdateKey1(x0, y0, xp0, yp0, s)
    # P_seq = LASM2D(mu, x_0, y_0, m*n)
    # P = P_seq.reshape(I.shape)

    # Step 2
    # a = np.ceil((x0+y0+1)*(10**7)) % (m)
    # b = np.ceil((xp0+yp0+2)*(10**7)) % (n)
    # u = P[int(a), :]
    # v = P[:, int(b)]
    # up = np.ceil(u*(10**14)) % (n)
    # vp = np.ceil(v*(10**14)) % (m)
    # up = up.astype(int)
    # vp = vp.astype(int)
    # Uniq(up)
    # Uniq(vp)
    B = np.zeros(I.shape, dtype='uint8')
    tmp = np.zeros(I.shape, dtype='uint8')
    for i in range(n):
        tmp[:, up[i]] = I[:, i]
    for i in range(m):
        B[vp[i], :] = tmp[i, :]
    
    # Step 3
    W = np.zeros(I.shape, dtype='uint8')
    for i in range(m):
        for j in range(n):
            W[i][j] = (m*n+(i+1)+(j+1)) % 256
    R = (B+W) % 256

    return R

def GetK(I):
    (m, n) = I.shape
    xp_0, yp_0 = UpdateKey2(x0, y0, xp0, yp0)
    K_seq = LASM2D(mu, xp_0, yp_0, m*n)
    K = K_seq.reshape(I.shape)
    K = np.ceil(K*(10**14)) % 256
    K = K.astype('uint8')
    
    return K

def GenerateRandomImage(I):
    M, N = I.shape
    s_I = Entropy(I)
    while True:
        Ip = np.zeros(M * N, dtype=I.dtype)
        cnt = np.zeros(256, dtype=int)
        for x in I.flatten():
            cnt[x] += 1
        np.random.shuffle(cnt)
        p = 0
        for x in range(256):
            for c in range(cnt[x]):
                Ip[p] = x
                p += 1
        np.random.shuffle(Ip)
        Ip = Ip.reshape(M, N)
        s_Ip = Entropy(Ip)
        if abs(s_I - s_Ip) < 1e-16:
            # print(s_I)
            # print(s_Ip)
            break

    return Ip

def break_K(I, up, vp):
    M, N = I.shape
    s_I = Entropy(I)

    X = np.zeros((N, M), dtype=int)
    p_start = N - 1

    total_round = 0
    while p_start >= 0:
        total_round += 1
        ### Random generate Ip with same entropy of I
        Ip = GenerateRandomImage(I)
        s_Ip = Entropy(I)
        ### Check the entropy of I and Ip are the same
        if abs(s_I - s_Ip) < 1e-16:
            print('Entropy Check: OK')
        else:
            print('Entropy Check: JIZZ')
            exit()
        ### Make equation system to solve
        R_Ip_true = GetR(Ip)
        R_Ip = GetRFromPermutation(Ip, up, vp)
        print(np.sum(R_Ip_true == R_Ip))
        if np.sum(R_Ip_true == R_Ip) != M * N:
            print('Wrong R')
            exit()

        C_Ip = FastEncrypt(Ip)
        A, B = RetrieveEquations(R_Ip, C_Ip)
        ### Gaussian Elimination
        A, B, X, p_stop = GaussianEliminationSolve(A, B, X, p_start)
        print(p_stop, X)
        p_start = p_stop
    X = X.T
    ### Print message when finished
    print('break_K finished, took {:d} rounds GE to solve.'.format(total_round))

    ### Check X equals to K
    K = GetK(I)
    print(K)
    print(np.sum(X == K))

    return X


if __name__ == '__main__':
    ### Load origin image I
    I = cv2.imread('img/test_lena_256.bmp', cv2.IMREAD_GRAYSCALE)
    I = cv2.resize(I, (260, 260))
    M, N = I.shape
    s_I = Entropy(I)


    X = np.zeros((N, M), dtype=int)
    p_start = N - 1

    total_round = 0
    while p_start >= 0:
        total_round += 1

        ### Random generate Ip with same entropy of I
        Ip = GenerateRandomImage(I)
        s_Ip = Entropy(I)
        ### Check the entropy of I and Ip are the same
        if abs(s_I - s_Ip) < 1e-16:
            print('Entropy Check: OK')
        else:
            print('Entropy Check: JIZZ')
            exit()
        
        R_Ip = GetR(Ip)
        C_Ip = FastEncrypt(Ip)
        A, B = RetrieveEquations(R_Ip, C_Ip)
        # print(R_Ip)
        # print(C_Ip)
        # print(A.shape)
        # print(A)
        # print(B.shape)
        # print(B)
        # exit()

        A, B, X, p_stop = GaussianEliminationSolve(A, B, X, p_start)
        # print(A)
        # print(B)
        print(p_stop, X)
        p_start = p_stop

    print('Finished ..., take {:d} round.'.format(total_round))
    print(X)
    K = GetK(Ip)
    print(K)
    print(np.sum(X.T == K))