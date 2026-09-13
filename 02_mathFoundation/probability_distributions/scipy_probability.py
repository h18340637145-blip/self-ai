from scipy import stats
import numpy as np

normal = stats.norm(loc = 0, scale = 1)
samples = normal.rvs(size = 1000)

print(f"mean = {np.mean(samples)} std = {np.std(samples)}")
print(f" p(x<1.96) = {normal.cdf(1.96)}")

logits = np.array([1.0, 2.0, 3.0])
from scipy.special import softmax, log_softmax
probs = softmax(logits)
log_probs = log_softmax(logits)
print(f"probs = {probs}")
print(f"log_probs = {log_probs}")
