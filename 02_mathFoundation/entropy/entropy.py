import math
# 每个事件的信息量
# I(x) = -log2(p)¬
def information_content(p, base=2):
    if p <= 0 or p > 1:
        return float('inf')
    return -math.log(p, base)

# 整个概率分布的平均信息量== Shannon 熵
# H(X) = Σ p(x) * I(x) = - Σ p(x) * log2(p(x))
def entropy(probs, base=2):
    # 计算熵
    return sum(
        p * information_content(p, base)
        for p in probs  
    )

# 公平的硬币案例
# 偏置的硬币案例
# 公平的骰子案例
fair_coin = [0.5, 0.5]
baised_coin = [0.9, 0.1]
fair_dir = [1/6] * 6


# 交叉熵和kl散度  H(P,Q)=−i∑​pi​logqi​
# 事件实际上按照 P 发生，但我们用模型 Q 的概率去评价这些事件有多意外。 这就是交叉熵的意义。
def cross_entropy(p, q, base=2):
    total = 0.0
    for pi, qi in zip(p, q):
        if pi > 0:
            if qi <= 0:
                return float('inf')
            total += pi * (-math.log(qi) / math.log(base)) # information content of qi
    return total

# KL散度
def kl_divergence(p, q, base=2):
    return cross_entropy(p, q, base) - entropy(p, base)

true_list = [0.7, 0.2, 0.1]
good_model = [0.6, 0.25, 0.15]
bad_model = [0.1, 0.1, 0.8]



if __name__ == "__main__":
    # 每个结果的信息量都是1   所以 0.5*1 + 0.5*1 = 1
    print("Entropy of fair coin:", entropy(fair_coin))

    # 偏置的硬币信息量较低，因为结果更可预测  0.468  可预测说明不混乱 熵就低
    print("Entropy of biased coin:", entropy(baised_coin))
    # 公平的骰子信息量较高，因为结果更不确定  2.585  不确定说明混乱 熵就高
    print("Entropy of fair die:", entropy(fair_dir))

    # 交叉熵和KL散度示例 
    # true list 的熵以及与不同模型的交叉熵和KL散度
    print(f"Entropy of true dist:     {entropy(true_list):.4f} bits")
    # good_model 为什么交叉熵比较低 因为它的概率分布更接近 true_list，预测更准确，信息量更低
    print(f"CE (good model):          {cross_entropy(true_list, good_model):.4f} bits")
    # bad_model 为什么交叉熵比较高 因为它的概率分布与 true_list 差距较大，预测不准确，信息量更高
    #  P = [0.7, 0.2, 0.1] 真实世界-----坏模型Q = [0.1, 0.1, 0.8] 非常明显
    print(f"CE (bad model):           {cross_entropy(true_list, bad_model):.4f} bits")
    # 
    print(f"KL divergence (good):     {kl_divergence(true_list, good_model):.4f} bits")
    print(f"KL divergence (bad):      {kl_divergence(true_list, bad_model):.4f} bits")  

'''
Entropy of true dist:     1.1568 bits
CE (good model):          1.1896 bits
CE (bad model):           3.0219 bits
KL divergence (good):     0.0328 bits
KL divergence (bad):      1.8651 

真实世界 P
[0.7, 0.2, 0.1]
        ↓
真实世界自己有多少不确定性？
        ↓
H(P)
1.1568 bits
        ↓
模型 Q 来预测这个世界
        ↓
H(P,Q)
        ↓
        ├── good model: 1.1896
        │
        └── bad model:  3.0219
'''

# 交叉熵作为分类损失
def softmax(logits):
    max_logits = max(logits)
    exps = [math.exp(z - max_logits) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

def cross_entropy_loss(true_class, logits):
    probs = softmax(logits)
    return -math.log(probs[true_class])


logits = [2, 1, 0.1]
true_class = 0
probs = softmax(logits)
loss = cross_entropy_loss(true_class,logits)

print(f"logits: {logits}")
print(f"softmax {[f'{p:.4f}' for p in probs]}")
print(f"true class {true_class}")
print(f"cross entropy loss: {loss:.4f}")
print(f"perplexity: {math.exp(loss):.4f}")

# 交叉熵等于负对数似然
import random
random.seed(42)

n_samples = 1000
n_classes = 3
true_labels = [random.randint(0, n_classes - 1) for _ in range(n_samples)]
model_logits = [[random.gauss(0, 1) for _ in range(n_classes)] for _ in range(n_samples)]

ce_loss = sum(
    cross_entropy_loss(label, logits)
    for label, logits in zip(true_labels, model_logits)
) / n_samples

nll = -sum(
    math.log(softmax(logits)[label])
    for label, logits in zip(true_labels, model_logits)
) / n_samples

print(f"Cross entropy loss: {ce_loss:.4f}")
print(f"Negative log-likelihood: {nll:.4f}")


# 相互信息
def mutual_information(joint_probs, base=2):
    rows = len(joint_probs)
    cols = len(joint_probs[0])

    margin_x = [sum(joint_probs[i][j] for j in range(cols)) for i in range(rows)]
    margin_y = [sum(joint_probs[i][j] for i in range(rows)) for j in range(cols)]

    mi = 0 
    for i in range(rows):
        for j in range(cols):
            pxy = joint_probs[i][j]
            if pxy > 0:
                mi += pxy * math.log(pxy / (margin_x[i] * margin_y[j]), base)
    return mi

independent = [[0.25, 0.25], [0.25, 0.25]]
dependent = [[0.45, 0.05], [0.05, 0.45]]

print(f"Mutual information (independent): {mutual_information(independent):.4f} bits")
print(f"Mutual information (dependent): {mutual_information(dependent):.4f} bits")