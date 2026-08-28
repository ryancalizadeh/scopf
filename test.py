import numpy as np

# 1. Define line impedances (Z = R + jX) in per unit (p.u.)
# Using realistic values where Resistance (R) is non-zero
z12 = 0.02 + 0.08j
z13 = 0.03 + 0.12j
z23 = 0.025 + 0.10j

# 2. Calculate line admittances (y = 1 / Z)
y12 = 1 / z12
y13 = 1 / z13
y23 = 1 / z23

# 3. Initialize a 3x3 complex NumPy ndarray
Ybus = np.zeros((3, 3), dtype=complex)

# 4. Populate Diagonal elements 
# Formula: Y_ii = sum of all admittances connected to bus i
Ybus[0, 0] = y12 + y13  # Bus 1
Ybus[1, 1] = y12 + y23  # Bus 2
Ybus[2, 2] = y13 + y23  # Bus 3

# 5. Populate Off-diagonal elements
# Formula: Y_ij = Y_ji = -y_ij (negative of the connecting admittance)
Ybus[0, 1] = Ybus[1, 0] = -y12  # Between Bus 1 and 2
Ybus[0, 2] = Ybus[2, 0] = -y13  # Between Bus 1 and 3
Ybus[1, 2] = Ybus[2, 1] = -y23  # Between Bus 2 and 3

# Optional: Format NumPy print output for easier reading
np.set_printoptions(precision=3, suppress=True)

print("3-Bus Ybus Matrix (in p.u.):\n")
print(Ybus)