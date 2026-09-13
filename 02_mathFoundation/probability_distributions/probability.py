import math
import random

# 计算阶乘的函数
def factorial(n):
    res = 1 
    for i in range(1, n + 1): # 从 1 到 n 依次相乘
        res *= i
    return res

# 计算组合数的函数
def combination(n, k):
    return factorial(n) // (factorial(k) * factorial(n - k))
# 计算条件概率的函数
def conditional_probability(p_a_and_b, p_b):
    return p_a_and_b / p_b

# 示例：计算在已知牌是正面朝上的情况下是国王的条件概率 p(king | face)  人头牌4*3=12  是人头牌且是国王的牌有4张
p_king_given_face = conditional_probability(p_a_and_b=4/52, p_b=12/52) # 计算在已知牌是正面朝上的情况下是国王的条件概率
print(f"p_king_given_face = {p_king_given_face}")  # 输出条件概率



# 构建pmf 和 pdf  

# 不同分布的概率质量函数 (PMF)
# 伯努利分布的概率质量函数 (PMF)
def bernoulli_pmf(k, p):
    return p if k == 1 else (1 - p)

# 多项分布的概率质量函数 (PMF)
def categorical_pmf(k, p_list):
    return p_list[k]

# 泊松分布的概率质量函数 (PMF)
def poisson_pmf(k, lam):
    return (lam ** k) * math.exp(-lam) / factorial(k)


# 连续分布的概率密度函数 (PDF)
# 正态分布的概率密度函数 (PDF)
def normal_pdf(x, mu, sigma):
    return (1 / (sigma * math.sqrt(2 * math.pi))) * math.exp(-0.5 * ((x - mu) / sigma) ** 2)

# uniform 分布的概率密度函数 (PDF)
# x 在区间 [a, b] 内的概率密度为 1/(b-a)，否则为 0
def uniform_pdf(x, a, b):
    if a <= x <= b:
        return 1 / (b - a)
    else:
        return 0


# 3 期望值和方差
def expected_value(values, probabilities):
    return sum(value * prob for value, prob in zip(values, probabilities))

def variance(values, probabilities):
    mean = expected_value(values, probabilities)
    return sum(prob * (value - mean) ** 2 for value, prob in zip(values, probabilities))


die_values = [1, 2, 3, 4, 5, 6]
die_probabilities = [1/6] * 6

mu = expected_value(die_values, die_probabilities)
var = variance(die_values, die_probabilities)

print(f"die_expected_value = {mu}")
print(f"die_variance = {var}")

# 从分布中抽样
def sample_bernoulli(p, n = 1):
    return [1 if random.random() < p else 0 for _ in range(n)]

# 从多项分布中抽样
def sample_categorical(p_list, n = 1):
    cumulative = []
    total = 0
    for p in p_list:
        total += p
        cumulative.append(total)
    samples = []
    for _ in range(n):
        r = random.random()
        for i, c in enumerate(cumulative):
            if r < c:
                samples.append(i)
                break
    return samples
# 从正态分布中抽样（使用 Box-Muller 变换）
def sample_normal_box_muller(mu, sigma, n=1):
    samples = []
    for _ in range(n):
        u1 = random.random()
        u2 = random.random()
        z = math.sqrt(-1 * math.log(u1)) * math.cos(2 * math.pi * u2)
        samples.append(mu + sigma * z)
    return samples

# softmax 和对数概率
def softmax(logits):
    max_logit = max(logits) # 为了数值稳定性，减去最大值
    shifted = [z - max_logit for z in logits] # softmax 的性质是减去一个常数不会改变输出的相对比例
    exps = [math.exp(z) for z in shifted]
    total = sum(exps)
    return [e / total for e in exps]

# 对数概率的 softmax (log-softmax)
def log_softmax(logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    log_sum_exp = math.log(sum(math.exp(z) for z in shifted))
    return [z - log_sum_exp for z in logits]

def cross_entropy_loss(logits, target_index):
    log_probs = log_softmax(logits)
    return -log_probs[target_index]

# 中心极限定理演示
def demonstrate_central_limit_theorem(dist_fn, n_samples, n_averages):
    averages = []
    for _ in range(n_samples):
        samples = [dist_fn() for _ in range(n_samples)]
        averages.append(sum(samples) / n_averages)
    return averages

# 可视化
import matplotlib.pyplot as plt

xs = [mu + var ** 0.5 * (i - 500) / 100 for i in range(1000)]
ys = [normal_pdf(x, mu, var ** 0.5) for x in xs]
plt.plot(xs, ys)

def plot_histogram(data, bins=30, title="Histogram", xlabel="Value", ylabel="Frequency"):
    plt.hist(data, bins=bins, edgecolor='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

