# 1. Safe Feature Scaling
- **`x.min(axis=0) / x.max(axis=0)`**: Computes the min/max per feature (column). If you don't use axis=0, large features (like income) will completely overshadow small features (like age).
- **`+ 1e-8 (Epsilon)`**: Adding a tiny number to the denominator prevents Division by Zero errors. If a feature has the exact same value in every row, max - min equals 0, which would instantly crash an automated pipeline.

# 2. Reproducible Experimentation
- **`np.random.seed(42)`**: Locks the random number generator. If your model's accuracy changes tomorrow, you need to know it was your code that caused it, not just a lucky/unlucky random data split.
- **`np.random.permutation()`**: Randomly shuffles row indices. Data is often sorted by date or ID; shuffling ensures your training, validation, and test sets all contain a healthy, randomized mix of data.

# 3. High-Performance I/O
- **`np.save() / np.load()`**: Saves arrays directly to binary .npy files. This is massively faster to read and write than parsing raw .csv files, saving hours of load time when dealing with gigabytes of pre-processed ML features.

# 4. Automated Sanity Checks
- **`np.isnan().any() / np.isinf().any()`**: Neural networks immediately crash or output garbage if they encounter a NaN (Not a Number) or Infinity. These checks serve as an alarm system before training starts.
- **`np.nan_to_num()`**: Automatically cleans dirty data by replacing NaNs with 0.0 and infinities with extremely large finite numbers. This ensures your production pipeline won't fail at 3 AM due to a single corrupted row of data.