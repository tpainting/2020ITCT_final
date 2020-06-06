import math
import numpy as np
from scipy.stats import entropy

# secret key
x0, y0, xp0, yp0 = 0.0056, 0.3678, 0.6229, 0.7676
mu = 0.8116

def LASM2D(mu, x0, y0, ret_num, skip_num=200):
    iter_num = ret_num//2+ret_num%2
    xi = x0
    yi = y0
    ret_seq = []
    for i in range(skip_num+iter_num):
        xi = math.sin(math.pi*mu*(yi+3)*xi*(1-xi))
        yi = math.sin(math.pi*mu*(xi+3)*yi*(1-yi))
        if i >= skip_num:
            ret_seq.append(xi)
            ret_seq.append(yi)
    ret_seq = ret_seq[:ret_num]
    return np.array(ret_seq)

def Entropy(seq, size=256):
    # grayscale
    seq = seq.flatten()
    prob = np.zeros(size)
    for pixel in seq:
        prob[pixel] += 1
    prob /= len(seq)
    return entropy(prob, base=2)

def UpdateKey1(x0, y0, xp0, yp0, s):
    x_bar0 = (x0+(s+1)/(s+xp0+yp0+1))%1
    y_bar0 = (y0+(s+2)/(s+xp0+yp0+2))%1
    return x_bar0, y_bar0

def UpdateKey2(x0, y0, xp0, yp0):
    xp_bar0 = (xp0+(1/(x0+y0+1)))%1
    yp_bar0 = (yp0+(2/(x0+y0+2)))%1
    return xp_bar0, yp_bar0

def Uniq(seq):
    now_set = set()
    min_num = 0
    for i, s in enumerate(seq):
        if s not in now_set:
            now_set.add(s)
            if s == min_num:
                while min_num in now_set:
                    min_num += 1
        else:
            seq[i] = min_num
            now_set.add(min_num)
            while min_num in now_set:
                min_num += 1

if __name__ == '__main__':
    import cv2
    img = cv2.imread("./img/test_lena_256.bmp", cv2.IMREAD_GRAYSCALE)
    print(Entropy(img))
