import numpy as np

# Replace 'path_to_file.npy' with the actual path to your .npy file
file_path = 'ddad_intrinsics.npy'

# Load the array from the .npy file
data = np.load(file_path)

# Print the array
print(data)
