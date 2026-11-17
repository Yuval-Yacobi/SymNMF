# SymNMF Clustering Engine: C & Python Integration

A high-performance implementation of the **SymNMF** (Symmetric Non-negative Matrix Factorization) clustering algorithm, benchmarked against **K-means** using the silhouette score.

To achieve optimal performance for heavy matrix computations, the core algorithmic engine is implemented in **C**, memory-optimized, and exposed as a **Python Extension module** using the **Python C API**. This allows users to enjoy Python's simple frontend while leveraging C's execution speed.

## Tech Stack & Key Features
* **Languages:** C, Python 3
* **Integration:** Python C API (C Extensions for Python)
* **Libraries:** NumPy, scikit-learn
* **Skills Demonstrated:** Low-level memory management, Algorithm optimization, Full lifecycle build (Makefiles, setup.py).
---

## Table of Contents
- [Mathematical Background](#mathematical-background)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Building](#building)
- [Usage](#usage)
- [Input and Output Format](#input-and-output-format)
- [Error Handling](#error-handling)
- [Implementation Notes](#implementation-notes)


---

## Mathematical Background

Given $N$ points $x_1,\dots,x_N \in \mathbb{R}^d$, the algorithm runs in four steps:

1. **Similarity matrix** $A \in \mathbb{R}^{N\times N}$:

   $$a_{ij} = \exp\left(-\frac{\|x_i - x_j\|^2}{2}\right) \quad (i \ne j), \qquad a_{ii} = 0$$

2. **Diagonal degree matrix** $D$:

   $$d_{ii} = \sum_{j=1}^{N} a_{ij}$$

3. **Normalized similarity matrix** $W$:

   $$W = D^{-1/2} A D^{-1/2}$$

4. **Find $H_{N\times k} \ge 0$** that minimizes $\|W - HH^T\|_F^2$:
   - **Initialization:** uniform random values in $[0,\ 2\sqrt{m/k}]$, where $m$ is the mean of all entries of $W$.
   - **Update rule** (with $\beta = 0.5$):

     $$H_{ij}^{(t+1)} \leftarrow H_{ij}^{(t)}\left(1 - \beta + \beta \cdot \frac{(WH^{(t)})_{ij}}{(H^{(t)}H^{(t)T}H^{(t)})_{ij}}\right)$$

   - **Convergence:** stop when $\|H^{(t+1)} - H^{(t)}\|_F^2 < \epsilon = 10^{-4}$ or after 300 iterations.

**Cluster assignment:** each point (a row of $H$) is assigned to the cluster (column) with its highest association score, i.e. `argmax` over the row.

---

## Project Structure

| File | Description |
|------|-------------|
| `symnmf.py` | Main Python interface. Parses arguments, loads the data, initializes $H$ and calls the C extension. |
| `symnmf.c` | C implementation: file reading, `sym`, `ddg`, `norm`, the $H$ update loop, and the `main` function of the standalone program. |
| `symnmf.h` | Header file with the data structures (`cord`, `vector`) and the prototypes of all functions used by `symnmfmodule.c`. |
| `symnmfmodule.c` | Python C API wrapper that exposes `sym`, `ddg`, `norm` and `symnmf` to Python. |
| `analysis.py` | Compares SymNMF to K-means (implementation from HW1) using `silhouette_score`. |
| `setup.py` | Build script that produces the Python extension (`symnmfmodule*.so`). |
| `Makefile` | Builds the `symnmf` executable with the required compilation flags. |

---

## Requirements

- `gcc`
- Python 3 with `setuptools`
- `numpy`
- `scikit-learn` (needed only for `analysis.py`)

```bash
pip install numpy scikit-learn
```

---

## Building

### Standalone C program
```bash
make
```
Compilation uses `-ansi -Wall -Wextra -Werror -pedantic-errors` and completes with no errors or warnings.

To remove build files:
```bash
make clean
```

### Python extension
```bash
python3 setup.py build_ext --inplace
```
This creates `symnmfmodule.*.so` in the current directory, which `symnmf.py` and `analysis.py` import.

---

## Usage

### `symnmf.py`
```bash
python3 symnmf.py <k> <goal> <file_name.txt>
```

| Argument | Meaning |
|----------|---------|
| `k` | Number of clusters. An integer, `1 ≤ k < N`. |
| `goal` | One of: `symnmf`, `sym`, `ddg`, `norm` |
| `file_name` | Path to the input file (`.txt`) |

| `goal` | Output |
|--------|--------|
| `symnmf` | The final matrix $H$ ($N \times k$) |
| `sym` | The similarity matrix $A$ |
| `ddg` | The diagonal degree matrix $D$ |
| `norm` | The normalized similarity matrix $W$ |

> Note: `k` must be passed for `sym`, `ddg` and `norm` as well, but it does not affect the computation.

Example:
```bash
$ python3 symnmf.py 2 symnmf input_1.txt
0.0600,0.0100
0.0100,0.0500
0.0100,0.0400
0.0200,0.0400
0.0500,0.0200
```

### C program
```bash
./symnmf <goal> <file_name.txt>
```
Here `goal` can only be `sym`, `ddg` or `norm`.

Example:
```bash
$ ./symnmf sym input_1.txt
0.0000,0.0447,0.0456,...
0.0447,0.0000,0.2871,...
...
```

### `analysis.py`
```bash
python3 analysis.py <k> <file_name.txt>
```
Runs both SymNMF and K-means on the same data and prints the silhouette score of each:
```bash
$ python3 analysis.py 5 input_k5_d7.txt
nmf: 0.1162
kmeans: 0.1147
```
A higher score indicates better-defined clusters.

---

## Input and Output Format

**Input:** a text file where each line is one point, with coordinates separated by commas:
```
1.2,3.4,5.6
7.8,9.0,1.2
...
```
All points are assumed to be distinct.

**Output:** each row of the matrix is printed on its own line, values are comma-separated, and every value is formatted to 4 decimal places (`%.4f`).

---

## Error Handling

On any error (wrong number of arguments, invalid `goal`, invalid `k`, failure to read the file, memory allocation failure, etc.) the program prints:
```
An Error Has Occurred
```
and terminates.

---

## Implementation Notes

- **Reading input in C:** the file is first read into a linked list of vectors (`vector` → `cord`), since the number of points and the dimension are not known in advance. The list is then copied into a 2D matrix.
- **Matrix allocation:** each matrix is allocated as one contiguous block of memory plus an array of row pointers. This makes memory access more efficient and freeing simpler (`free_matrix`).
- **Randomness:** `symnmf.py` sets `np.random.seed(1234)` at the top of the code, and $H$ is initialized with `np.random.uniform`, so results are reproducible.
- **Division of work between languages:** Python handles argument parsing, data loading and $H$ initialization; C handles all matrix computations and the optimization loop.
- **K-means** (in `analysis.py`): the initial centroids are the first $k$ points. Convergence uses $\epsilon = 10^{-4}$ or at most 300 iterations, and empty clusters are handled.
- **Memory management:** all allocated memory is freed, including on error paths.
- **Types:** `double` in C and `float` in Python for all vector elements.


