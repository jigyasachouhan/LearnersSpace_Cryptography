import numpy as np
from math import gcd

# Sample plaintext and ciphertext for testing
plaintext = "PRISHACHOUHAN"
ciphertext = "HKMKQOCOICMOXYZ"

# Function for performing Gaussian elimination to solve linear systems of equations
def gaussian_elimination(a, b):
    size = len(a)  # Size of the square matrix
    augmented_matrix = np.hstack((a, b)).astype(float)  # Combine A and B into an augmented matrix
    
    # Forward Elimination to create an upper triangular matrix
    for i in range(size):
        if augmented_matrix[i][i] == 0.0:  # Swap rows if the diagonal element is zero
            for j in range(i + 1, size):
                if augmented_matrix[j][i] != 0.0:
                    augmented_matrix[[i, j]] = augmented_matrix[[j, i]]
                    break

        augmented_matrix[i] = augmented_matrix[i] / augmented_matrix[i][i]  # Normalize the pivot row

        for j in range(i + 1, size):  # Eliminate the entries below the pivot
            augmented_matrix[j] -= augmented_matrix[i] * augmented_matrix[j][i]

    # Backward Elimination to create a diagonal matrix
    for i in range(size - 1, -1, -1):
        for j in range(i - 1, -1, -1):
            augmented_matrix[j] -= augmented_matrix[i] * augmented_matrix[j][i]
            
    # Extracting the solution matrix
    solution_matrix = augmented_matrix[:, size:]
    return solution_matrix

# Function to convert text into a numerical matrix and pad it if necessary
def pad_matrix(text):
    text_length = len(text)
    if text_length % 3 != 0:
        text += "X" * (3 - text_length % 3)  # Pad with 'X' to make length a multiple of 3
    text_matrix = np.array([ord(char) - 65 for char in text]).reshape(-1, 3)  # Convert characters to numbers and reshape
    return text_matrix

# Function to find the key matrix using Gaussian elimination
def key_matrix_search(plain_mat, cipher_mat):
    column_count = len(cipher_mat)
    found = False
    for i in range(column_count):
        for j in range(i + 1, column_count):
            for k in range(j + 1, column_count):
                # Forming a submatrix using columns i, j, k
                sub_matrix = np.column_stack((plain_mat[i], plain_mat[j], plain_mat[k])).astype(float)
                det = round(np.linalg.det(sub_matrix))  # Calculate the determinant
                if gcd(det, 26) == 1:  # Check if the determinant is coprime with 26
                    # Calculate the cofactor matrix
                    cofactor_mat = np.matrix([[(sub_matrix[(row+1)%3][(col+1)%3]*sub_matrix[(row+2)%3][(col+2)%3]-sub_matrix[(row+1)%3][(col+2)%3]*sub_matrix[(row+2)%3][(col+1)%3]) for row in range(3)] for col in range(3)]).astype(int) % 26
                    temp_key = np.column_stack((cipher_mat[i], cipher_mat[j], cipher_mat[k])).astype(int) % 26
                    temp_key = np.matmul(temp_key, cofactor_mat)  # Multiply with cofactor matrix
                    temp_key *= pow(det, -1, 26)  # Multiply by the modular inverse of the determinant
                    temp_key %= 26  # Reduce modulo 26
                    found = True
                    break
            if found:
                break
        if found:
            break
    if found:
        return temp_key
    else:
        return None

# Function to find the key matrix by brute force
def brute_force_key_search(plain_mat, cipher_mat, row_idx):
    potential_keys = []
    result_vector = cipher_mat.T[row_idx]
    for col in range(len(cipher_mat)):
        for idx in range(3):
            if plain_mat[col][idx] % 2 == 0 or plain_mat[col][idx] % 13 == 0:
                continue  # Skip columns with values that are factors of 26
            if idx == 0:
                first_row, second_row = 1, 2
            elif idx == 1:
                first_row, second_row = 0, 2
            else:
                first_row, second_row = 0, 1
            for i in range(26):
                for j in range(26):
                    # Calculate the third key value using modular arithmetic
                    k = ((cipher_mat[col][row_idx] - i * plain_mat[col][first_row] - j * plain_mat[col][second_row]) * pow(int(plain_mat[col][idx]), -1, 26)) % 26
                    if idx == 0:
                        key_vec = np.array([k, i, j])
                    elif idx == 1:
                        key_vec = np.array([i, k, j])
                    else:
                        key_vec = np.array([i, j, k])
                    # Check if the key vector produces the correct result vector
                    if np.array_equal(np.matmul(key_vec, plain_mat.T) % 26, result_vector):
                        potential_keys.append(key_vec)
            return potential_keys
    # Brute force search through all possible values if the key was not found above
    for i in range(26):
        for j in range(26):
            for k in range(26):
                key_vec = np.array([i, j, k])
                if np.array_equal(np.matmul(key_vec, plain_mat.T) % 26, result_vector):
                    potential_keys.append(key_vec)
    return potential_keys

 # Convert the plaintext and ciphertext into numerical matrices
plain_matrix = pad_matrix(plaintext)
cipher_matrix = pad_matrix(ciphertext)

# Try to find the key matrix using Gaussian elimination
key_matrix = key_matrix_search(plain_matrix, cipher_matrix)
if key_matrix is not None:
    print("Key Matrix:\n", key_matrix)
    key_matrix = np.array(key_matrix)  # Convert to ndarray for rounding
    key_string = "".join([chr(int(round(char)) + 65) for char in key_matrix.flatten()])
    print("Key String:", key_string)
    exit()

# If key matrix is not found, perform brute force search
potential_key_matrices = []
for row in range(3):
    potential_key_matrices.append(brute_force_key_search(plain_matrix, cipher_matrix, row))

# Print all potential key matrices found
for key_1 in potential_key_matrices[0]:
    for key_2 in potential_key_matrices[1]:
        for key_3 in potential_key_matrices[2]:
            combined_key = np.column_stack((key_1, key_2, key_3))
            print("Potential Key Matrix:\n", combined_key)

