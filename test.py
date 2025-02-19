import numpy as np

arr1 = np.random.rand(2)      # Shape (5,)
arr2 = np.random.rand(2, 3)   # Shape (5,7)

print("arr1")
print(arr1)
print()

print("arr2")
print(arr2)


arr1_reshaped = arr1.reshape(-1, 1)  # Convert (5,) -> (5,1)

stacked = np.hstack((arr1_reshaped, arr2))  # Stacking along axis 1

print(stacked.shape)  # Output: (5, 8)

print(stacked)