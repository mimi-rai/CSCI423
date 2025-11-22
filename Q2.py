import argparse
import bisect
import random
import sys
import numpy as np
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("filename")
parser.add_argument("seed", type=int)
parser.add_argument("N", type=int)
args = parser.parse_args()

# ========================================================================================================
# PART A
# ========================================================================================================

# read points from disk file
x = []
y = []

# read from .dat file given as command line argument and add data points
filename = args.filename
N = args.N

try:
    with open(filename, 'r') as f:
        for line in f:
            try:
                values = line.strip().split()
                assert len(values) == 2, f"expected a line to have two values, but it has {len(values)} values: {values}"
                x.append(float(values[0]))
                y.append(float(values[1]))
            except (ValueError, AssertionError) as e:
                print(f"Invalid data in file: {line.strip()} - {e}")
except FileNotFoundError:
    print(f"File not found: {filename}")
    sys.exit(1)


# normalizes piecewise curve to 1 by scaling y_i (creating PDF)
# find total area
total_area = np.trapezoid(y, x)
# divide each y by current area
y = [(float)(y_i / total_area) for y_i in y]

# CHECK: total area is 1 with normalized y
# print(np.trapezoid(y,x))

# CHECK: plot normalized graph
# plt.plot(x,y)
# plt.scatter(x,y)
# plt.title("Normalized Data Plot")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.show()


# ========================================================================================================
# PART B
# ========================================================================================================

# calculate CDF values
F = [y[0]]

for i in range(1, len(x)):
    # calculate current area between j-1 and j (j+1 is excluded)
    area = (float)(np.trapezoid(y[i-1:i+1], x[i-1:i+1]))
    F.append(F[-1] + area)

# CHECK: plot CDF graph
# plt.plot(range(len(x)),F)
# plt.scatter(range(len(x)),F)
# plt.title("CDF Plot")
# plt.xlabel("i")
# plt.ylabel("F")
# plt.show()  


# ========================================================================================================
# PART C
# ========================================================================================================

def F_i(A, B, C, x):
    return A * x**2 + B * x + C

A = []
B = [] 
C = []

# calculate all coefficients to use to calculate script F
for ind in range(1, len(x)):
    # calc slope = line segment between (x[i-1], y[i-1]) to (x[i], y[i])
    mi = (y[ind] - y[ind-1]) / (x[ind] - x[ind-1])
    
    # calc A = (m/2)
    A.append(mi / 2)

    # calc B = y[i-1] - mi * x[i-1]
    B.append(y[ind-1] - mi * x[ind-1])

    # calc C = (m/2) * x[i-1]^2 - y[i-1] * x[i-1]
    C.append((mi / 2) * x[ind-1]**2 - y[ind-1] * x[ind-1])

# second for loop to calculate the script F 
ind = 0
offset = 0
scriptF = []
pieceF = [] # piecewise CDF
dense_x = np.linspace(x[0], x[-1], 1000)
for curr_i in dense_x:
    if curr_i > x[ind+1]:
        ind += 1
        offset += scriptF[-1]

    scriptF.append(F_i(A[ind], B[ind], C[ind], curr_i))
    pieceF.append(offset + scriptF[-1])

# CHECK: plot graph
# plt.plot(dense_x,scriptF)
# plt.axhline(0, color='black')
# plt.title("Per Segment Quadrative Curves")
# plt.xlabel("x")
# plt.ylabel("Script F")
# plt.show()  

# CHECK: plot graph for piecewise CDF
# plt.plot(dense_x,pieceF)
# plt.axhline(0, color='black')
# plt.title("Piecewise CDF")
# plt.xlabel("x")
# plt.ylabel("F")
# plt.show()


# ========================================================================================================
# PART D
# ========================================================================================================

random.seed(args.seed)
inv_x = []
# test check all variates
u_list = []
u_prime_list = []
# loop to N
for _ in range(N):
    # u = random variate
    u = random.random()
    u_list.append(u)

    # binary search for left-most segment index s.t. F(x_ind) > u
    ind = bisect.bisect_left(F, u) - 1
    print(f"{ind=}")

    u_prime = u - F[ind]
    u_prime_list.append(u_prime)

    a = A[ind]
    b = B[ind]
    c = C[ind] - u_prime

    # print(b**2 - 4*a*c)
    x1 = (-b + (b**2 - 4*a*c)**.5) / (2*a)
    x2 = (-b - (b**2 - 4*a*c)**.5) / (2*a)

    # select the x that actually falls in the range (x[i-1], x[i])
    inv_x.append(x1 if x1 >= x[ind-1] and x1 <= x[ind] else x2)

# CHECK: check inverse x
print("U List: ", u_list)
print("U Prime List: ", u_prime_list)
print("Inverse X: ", inv_x)
# NOTE: Inverse X is not in the correct x range at the moment



# ========================================================================================================
# PART E
# ========================================================================================================

plt.hist(inv_x, density=True, bins=20, color="blue")
plt.plot(x,y, color="black", alpha=0.5)
plt.title("Binned and Expected PDF")
plt.xlabel("x")
plt.ylabel("f(x0)")
plt.show()
