import sys
import numpy as np
import symnmfmodule

np.random.seed(1234)

"""
Coordinates the full SymNMF process by calculating 
the normalized similarity and initializing H.
The matrix H is initialized with values from a uniform 
distribution in the range [0, 2*sqrt(m/k)].
Parameter points: Input data points list.
Parameter N: Total number of data points.
Parameter k: Desired number of clusters.
Return: Final optimized H matrix or None.
"""
def run_symnmf(points, N, k):
    W_list = symnmfmodule.norm(points)
    if W_list is None:
        print("An Error Has Occurred")
        return None
    W_np = np.array(W_list)
    m = np.mean(W_np) 
    bound = 2 * np.sqrt(m / k)
    H_initial_np = np.random.uniform(0,bound, size=(N, k))
    H_initial_list = H_initial_np.tolist()
    final_H = symnmfmodule.symnmf(H_initial_list,W_list, N, k)
    if final_H is None:
        print("An Error Has Occurred")
        return None
    return final_H

"""
Iterates through the matrix and prints each row 
formatted to the required output standard.
Values are separated by commas and displayed 
with exactly four decimal places of precision.
Parameter matrix: List of lists to print.
"""
def print_matrix(matrix):
    for row in matrix:
        formatted_row = ",".join(format(cell, ".4f") for cell in row)
        print(formatted_row)

""" Parses command line interface arguments, loads data, and executes the specified SymNMF goal."""
def main():
    if (len(sys.argv) != 4):
        print("An Error Has Occurred")
        sys.exit(1)
    try:
        k = int(sys.argv[1])
        goal = sys.argv[2]
        file_path = sys.argv[3]
    except(ValueError, IndexError):
        print("An Error Has Occurred")
        sys.exit(1)
    valid_goals = ["symnmf", "sym", "ddg", "norm"]  # Validation: k must be an integer > 1, and goal must be valid
    if k < 1 or goal not in valid_goals:
        print("An Error Has Occurred")
        sys.exit(1)
    try:
        data = np.genfromtxt(file_path, delimiter=',') # Load N data points; genfromtxt ignores trailing empty lines
        if data.ndim == 1: 
            data = data.reshape(-1, 1)
        points = data.tolist()
        N = len(points)
    except Exception:
        print("An Error Has Occurred")
        sys.exit(1)
    if k >= N:
        print("An Error Has Occurred")
        sys.exit(1)
    if goal == "symnmf": # Execute logic based on the specified goal
        result = run_symnmf(points, N, k)
    elif goal == "sym":
        result = symnmfmodule.sym(points)
    elif goal == "ddg":
        result = symnmfmodule.ddg(points)
    elif goal == "norm":
       result = symnmfmodule.norm(points)
    if result is None:
        print("An Error Has Occurred")
        sys.exit(1)
    print_matrix(result) 

        
if __name__ == "__main__":
    main()