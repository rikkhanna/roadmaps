# Core NumPy Operations in ML

## 1. The Building Blocks

- **`np.array()`**: The foundation. Converts slow Python lists into high-speed C-arrays. Every piece of data in ML must be converted to an array.
- **`x.shape`**: The most important attribute! It tells you your dimensions (e.g., 2 rows, 2 columns). Most AI crashes happen because the shape of the data doesn't match the shape the model expects.
- **`np.zeros()`**: Creates a "blank canvas." Pre-allocating memory up front is drastically faster than appending items to a list in a loop.
- **`np.random.randn()`**: Generates matrices of random numbers. This is exactly how Neural Networks initialize their internal "weights" before they start learning.

## 2. Array Metadata

- **`x.dtype`**: The data type (like 32-bit vs 64-bit float). Crucial for optimizing memory on your Mac's M2 chip.
- **`x.ndim` & `x.size`**: Tells you the dimensions (1D vector, 2D matrix, 3D tensor) and the total number of elements. Used for sanity checks.

## 3. Reshaping (Formatting data for AI)

- **`x.reshape()`**: Reorganizes data without changing the values.
- **`x.flatten()`**: Smashes a grid into a single 1D line. You must do this before passing a 2D image into a classic ML algorithm like XGBoost.
- **`np.expand_dims()`**: Wraps your array in an extra dimension. If a model is trained on a "batch" of 32 images `(32, height, width)`, and you want to predict on just 1 image, you must use this to fake a batch size of 1 `(1, height, width)` or the model will fail.

## 4. Slicing & Math

- **`x[0, :]` / `x[:, 1]`**: Slicing. This is how you split an entire dataset to separate your input features (X) from your labels/answers (y).
- **`x * 2`, `x + 1`, `np.sqrt(x)`**: Broadcasting. Instead of writing a slow Python `for` loop to multiply every number, NumPy does the math on the entire array simultaneously in C. This is what makes NumPy incredibly fast.