import cv2
from encrypt import FastEncrypt
from break_K import GenerateRandomImage
from break_all import break_all



if __name__ == '__main__':
    I = cv2.imread('img/test_lena_256.bmp', cv2.IMREAD_GRAYSCALE)
    I = cv2.resize(I, (300, 300))
    cv2.imshow('Plaintext', I)
    cv2.moveWindow('Plaintext', 0, 0)
    cv2.waitKey()

    C = FastEncrypt(I)
    cv2.imshow('Ciphertext', C)
    cv2.moveWindow('Ciphertext', 500, 0)
    cv2.waitKey()

    A = GenerateRandomImage(I)
    cv2.imshow('Image with the same entropy as the Plaintext', A)
    cv2.moveWindow('Image with the same entropy as the Plaintext', 0, 400)
    cv2.waitKey()

    P = break_all(C, A)
    cv2.imshow('Image we broke from the ciphertext and random image', P)
    cv2.moveWindow('Image we broke from the ciphertext and random image', 500, 400)
    cv2.waitKey()
