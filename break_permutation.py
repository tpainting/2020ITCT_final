from lib import *
from IPython import embed
from math import log
import numpy as np
import cv2
import sys
import random
from encrypt import Encrypt
from time import time

up_ans = []
vp_ans = []

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

def break_permutation(A):
    # DONE break row permutation
    vp_guess = np.full(M, -1, dtype='int')
    B = Encrypt(A)
    #B, up_ans, vp_ans = Encrypt(A, ret_uv=1)
    cnt_A = get_cnt(A)

    for col in range(N):
        for i in range(256):
            x = np.where(cnt_A == cnt_A[i]-1)[0]
            if len(x) > 0:
                p = np.where(A == i)
                T1 = swap(A, (0, col), (p[0][0], p[1][0]))
                s1 = Entropy(T1)
                found = 0
                for xx in x:
                    T1[0][col] = xx
                    s2 = Entropy(T1)
                    if abs(s1-s2) < 1e-16:
                        found = 1
                        break
                if found: break

        for row in range(M):
            p = np.where(A == i)
            T = swap(A, (row, col), (p[0][0], p[1][0]))
            B = Encrypt(T)

            curr_cnt = cnt_A[i]
            for x in np.where(cnt_A == curr_cnt-1)[0]:
                T[row][col] = x

                C = Encrypt(T)
                D = C-B
                if sum(D[:,0] != 0) == 1:
                    print("HELL YEAAAH")
                    vp_guess[row] = np.where(D[:,0]!=0)[0][0]
                    break

            # wrong column
            if vp_guess[row] == -1: break
        # found
        if (vp_guess[row] != -1).all(): break

    
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
            for j in range(M):
                if j == off or j == off+256:
                    continue
                if any(D[j] != 0):
                    break
            v = first_not_zero(D[off])
            if v != -1:
                up_guess[i] = v
                break

    if (up_guess == -1).any() || (vp_guess == -1).any():
        return [], []
    return up_guess, vp_guess



 
# 115m44.706s
# 110m28.674s
# 102m11.166s
# 106m21.373s
# 105m27.148s
if __name__ == '__main__':
    random.seed(123456)
    N = 512
    M = 512
    cnt = np.zeros(256, dtype='int')
    A = np.array([[0]*N]*M, dtype='uint8')
    for i in range(M):
        for j in range(N):
            A[i][j] = random.randint(0, 100)
            cnt[A[i][j]] += 1

    '''
    W = np.zeros(A.shape, dtype='uint8')
    for i in range(M):
        for j in range(N):
            W[i][j] = (M*N+i+1+j+1) % 256
    '''

    clock = time()


    # DONE break row permutation
    vp_guess = np.full(M, -1, dtype='int')
    #B = Encrypt(A)
    B, up_ans, vp_ans = Encrypt(A, ret_uv=1)
    print(np.where(up_ans == 0))
    cnt_A = get_cnt(A)

    for col in range(N):
    #for col in range(65,66):
        print('col:', col)
        for i in range(256):
            x = np.where(cnt_A == cnt_A[i]-1)[0]
            if len(x) > 0:
                p = np.where(A == i)
                T1 = swap(A, (0, col), (p[0][0], p[1][0]))
                s1 = Entropy(T1)
                found = 0
                for xx in x:
                    T1[0][col] = xx
                    s2 = Entropy(T1)
                    if abs(s1-s2) < 1e-16:
                        found = 1
                        break
                if found: break

        for row in range(M):
            print('row:', row, i)
            p = np.where(A == i)
            T = swap(A, (row, col), (p[0][0], p[1][0]))
            B = Encrypt(T)

            curr_cnt = cnt_A[i]
            for x in np.where(cnt_A == curr_cnt-1)[0]:
                T[row][col] = x

                C = Encrypt(T)
                D = C-B
                if sum(D[:,0] != 0) == 1:
                    print("HELL YEAAAH")
                    vp_guess[row] = np.where(D[:,0]!=0)[0][0]
                    break

            # wrong column
            if vp_guess[row] == -1: break
        # found
        if (vp_guess[row] != -1).all(): break


    clock2 = time()

    #print(vp_guess)
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

    clock3 = time()

    #print(up_guess)
    if all(up_ans == up_guess):
        print("Oh yeaaaaa")

    if (vp_ans == vp_guess).all() and all(up_ans == up_guess):
        print("All set!!!")
    else:
        print("Something went wrong... :(")

    print('Break row time:', clock2-clock)
    print('Break col time:', clock3-clock2)
    #embed()
    


