from lib import *
from IPython import embed
from math import log
import numpy as np
import cv2
import sys
import random
#import encrypt

up_ans = []
vp_ans = []

def get_R(A):
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


def Encrypt(A):
    # Step 1
    (m, n) = A.shape
    s = Entropy(A)
    #print('s: %.40f\n' % s)

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

    global up_ans, vp_ans
    if len(up_ans) == 0:
        up_ans = up
    elif not all(up_ans == up):
        print("JIZZZ")
    if len(vp_ans) == 0:
        vp_ans = vp
    elif not all(vp_ans == vp):
        print("JIZZZ")



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

def swap(A, p1, p2):
    T = np.array(A)
    tmp = T[p1[0]][p1[1]]
    T[p1[0]][p1[1]] = T[p2[0]][p2[1]]
    T[p2[0]][p2[1]] = tmp
    return T

def first_not_zero(A):
    for i in range(len(A)):
        if A[i] != 0:
            return i
    return -1

def test_col(A):
    for i in range(A.shape[1]):
        x = sum(A[:, i] != 0)
        if x == 1:
            return np.where(A[:, i] == True)[0]
        elif x > 1:
            return -1
    return -2

def get_cnt(A):
    cnt = np.zeros(256, dtype='int')
    for i in A:
        for j in i:
            cnt[j] += 1
    return cnt


if __name__ == '__main__':
    random.seed(711222)
    N = 512
    M = 512
    cnt = np.zeros(256, dtype='int')
    v2p = [[] for i in range(256)]
    A = np.array([[0]*N]*M, dtype='uint8')
    for i in range(M):
        for j in range(N):
            A[i][j] = random.randint(0, 100)
            cnt[A[i][j]] += 1
            v2p[A[i][j]].append((i, j))


    W = np.zeros(A.shape, dtype='uint8')
    for i in range(M):
        for j in range(N):
            W[i][j] = (M*N+i+1+j+1) % 256


    vp_guess = np.full(M, -1, dtype='int')
    B = Encrypt(A)
    cnt_orig = get_cnt(A)
    # (0, 28)
    for col in range(N):
        print('col:', col)
        for row in range(M):
            end = 0
            for i in range(1):
                print(row, i)
                p = np.where(A == i)
                if len(p[0]) > 0:
                    T = swap(A, (row, col), (p[0][0], p[1][0]))
                    B = Encrypt(T)
                    cnt_T = get_cnt(T)

                    curr_cnt = cnt_T[i]
                    for x in np.where(cnt_T == curr_cnt-1)[0]:
                        T[row][col] = x

                        #embed()
                        C = Encrypt(T)
                        D = C-B
                        if sum(D[:,0] != 0) == 1:
                            print("HELL YEAAAH")
                            vp_guess[row] = np.where(D[:,0]!=0)[0][0]
                            end = 1
                            break
                        else:
                            pass
                            #print("NOOOOO")
                if end: break
            # wrong column
            if vp_guess[row] == -1: break
        if (vp_guess[row] != -1).all(): break


    #from vp_guess import VP_GUESS
    #vp_guess = np.array(VP_GUESS)
    #print(vp_guess)
    #print(vp_ans)
    if (vp_ans == vp_guess).all():
        print("Oh yeaaaaa")
               
   
    
    # DONE break column permutation
    up_guess = np.full(N, -1, dtype='int')
    B = Encrypt(A)
    for i in range(N):
        for off in range(M-256):
            r1 = np.where(vp_guess == off)[0][0]
            r2 = np.where(vp_guess == off+256)[0][0]
            T = swap(A, (r1, i), (r2, i))
            C = Encrypt(T)
            D = C-B
            #print(C-B, '\n')
            for j in range(M):
                if j == off or j == off+256:
                    continue
                if any(D[j] != 0):
                    #print("ERROR")
                    break
            v = first_not_zero(D[off])
            if v != -1:
                print(i, off)
                up_guess[i] = v
                break

    if all(up_ans == up_guess):
        print("Oh yeaaaaa")

    #embed()
    


