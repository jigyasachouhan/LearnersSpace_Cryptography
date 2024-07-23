import numpy
from math import sqrt

#function to pad plaintext with 'X' to make it a multiple of n
def pad_array_with_x(arr, n):
    length = len(arr)
    if length % n == 0:
        padded_array = arr
    else:
        padded_array = arr + 'X'*(n - length%n)
    return padded_array


key = input("Enter the key: ").upper().strip()
key = [ord(letter) - ord('A') for letter in key]
n = sqrt(len(key))      # n is the size of the key matrix
n = int(n)
plaintext = input("Enter the plaintext: ").upper().strip()    
plaintext = pad_array_with_x(plaintext, n)   # Pad the plaintext with 'X'
plaintext = [ord(letter) - ord('A') for letter in plaintext]   # Convert the plaintext to numbers
key = numpy.array(key).reshape(n,n)    # Reshape the key matrix
plaintext = numpy.array(plaintext).reshape(-1,n).T   # Transpose the plaintext matrix
ciphertext = numpy.dot(key, plaintext)%26    # Perform matrix multiplication
ciphertext = ciphertext.T
ciphertext = ciphertext.flatten()     # Flatten the matrix
ciphertext = [chr(i+ord('A')) for i in ciphertext]   # Convert the numbers to letters
ciphertext = ''.join(ciphertext)
print("Cipher Text: ", ciphertext)