import numpy as np
from hmmlearn import hmm

np.random.seed(42)

model = hmm.MultinomialHMM(n_components=3)
model.startprob_ = np.array([0.6, 0.3, 0.1])

print model.fit(np.array([[1], [2], [3], [1], [2], [0]]))
