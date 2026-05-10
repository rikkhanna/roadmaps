# pyrefly: ignore [missing-import]
import numpy as np

x = np.array([[1.0, 2.0], [3.0, 4.0]])
print(x.shape)

zeros = np.zeros((100, 10))
print(zeros)

rand = np.random.randn(2, 3)
print(rand)

print(x.dtype)
print(x.ndim)
print(x.size)
print(x)
# Reshaping critical for feeding models
print(x.reshape(1, 4))
print(x.flatten())
print(np.expand_dims(x, axis=0))

print(x[0, :])
print(x[:, 1])

print(x*2)
print(x+1)
print(np.sqrt(x))