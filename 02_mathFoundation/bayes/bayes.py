
# 贝叶斯定理计算
def bayes(prior,likelihood, false_positive_rate):
    evidence = likelihood * prior + false_positive_rate * (1 - prior) # 计算边际概率
    posterior = (likelihood * prior) / evidence # 计算后验概率
    return posterior

result = bayes(prior=0.01, likelihood=0.9, false_positive_rate=0.05)
print(f"Posterior probability: {result}")


# 朴素贝叶斯分类器
import math 
from collections import defaultdict

class NaiveBayes:
    def __init__(self, smoothing = 1):
        self.smoothing = smoothing # 平滑参数，用于处理零概率问题
        self.class_counts = defaultdict(int) # 记录每个类别的文档数量
        self.words_counts = defaultdict(lambda: defaultdict(int)) # 记录每个类别下每个单词的出现次数
        self.class_word_totals = defaultdict(int) # 记录每个类别下单词的总数
        self.vocab = set() # 记录所有出现过的单词

    # 训练朴素贝叶斯分类器
    def train(self, documents, labels):
        # 遍历每个文档及其对应的标签，更新各类统计信息
        for doc, label in zip(documents, labels):
            self.class_counts[label] += 1
            words = doc.lower().split()
            # 将文档拆分为单词，统计每个单词在该类别下的出现次数，并更新总词数和词汇表
            for word in words:
                self.words_counts[label][word] += 1
                self.class_word_totals[label] += 1
                self.vocab.add(word)

    # 预测新文档的类别
    def predict(self, document):
        words = document.lower().split()
        total_docs = sum(self.class_counts.values()) # 计算总文档数
        vocab_size = len(self.vocab) # 计算词汇表大小

        best_class = None
        best_score = float('-inf') # 初始化最佳得分为负无穷大

        for cls in self.class_counts:
            # 计算类别的先验概率
            prior = math.log(self.class_counts[cls] / total_docs) # 计算类别的先验概率的对数
            score = prior # 初始化该类别的得分为先验概率的对数
            for word in words:
                count = self.words_counts[cls][word] # 获取当前类别下该单词的出现次数
                total = self.class_word_totals[cls] # 获取当前类别下单词的总数
                # 计算单词在该类别下的条件概率的对数，并累加到该类别的得分中
                score += math.log((count + self.smoothing) / (total + self.smoothing * vocab_size)) # 计算单词在该类别下的条件概率的对数
            if score > best_score:
                best_score = score
                best_class = cls
        return best_class # 返回得分最高的类别

train_docs = [
    "win free money now",
    "free lottery ticket winner",
    "claim your prize today free",
    "urgent offer free cash",
    "congratulations you won free",
    "meeting tomorrow at noon",
    "project update attached",
    "can we schedule a call",
    "quarterly report review",
    "lunch on thursday sounds good",
    "team standup notes attached",
    "please review the pull request",
]

train_labels = [
    "spam", "spam", "spam", "spam", "spam",  # 垃圾邮件
    "ham", "ham", "ham", "ham", "ham", "ham", "ham",  # 非垃圾邮件
]

# 所以先验概率是：
# P(spam)=\frac{5}{12}
# P(ham)=\frac{7}{12}

# 开始统计 
print("开始统计")
# 例如 free 在垃圾邮件中出现很多次：
#  \(P(free|spam)\)? 会比较高
classifier = NaiveBayes()
classifier.train(train_docs, train_labels)

# 测试分类器
test_messages = [
    "free money waiting for you",
    "meeting rescheduled to friday",
    "you won a free prize",
    "please review the attached report",
]

for doc in test_messages:
    print(f"Document: '{doc}' => Predicted class: '{classifier.predict(doc)}'")

'''
"you won a free prize"
        ↓

分别假设：

它是 spam
↓
P(spam)
× P(you|spam)
× P(won|spam)
× P(a|spam)
× P(free|spam)
× P(prize|spam)

它是 ham
↓
P(ham)
× P(you|ham)
× P(won|ham)
× P(a|ham)
× P(free|ham)
× P(prize|ham)

        ↓

比较两个结果

        ↓

spam 得分更高

        ↓

预测 spam





模型根本“不理解” free、meeting、report 是什么意思。
它只是根据训练数据发现：
free → 经常和 spam 一起出现
meeting → 经常和 ham 一起出现
所以朴素贝叶斯本质上是：统计词和类别之间的概率关系
'''


def show_top_words(classifier, cls, n=5):
    vocab = len(classifier.vocab)
    total = classifier.class_word_totals[cls]
    probs = {}
    for word in classifier.vocab:
        count = classifier.words_counts[cls][word]
        probs[word] = (count + classifier.smoothing) / (total + classifier.smoothing * vocab)
    top_words = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:n]
    for word, prob in top_words:
        print(f"{word}: {prob:.4f}")
print("Top words for spam:")
show_top_words(classifier, "spam")
print("\nTop words for ham:")
show_top_words(classifier, "ham")
