import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import wordnet
import re

# 下载 'averaged_perceptron_tagger' 数据包
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')

# 数据加载和预处理
df = pd.read_csv('mbti_1.csv')
df = df.applymap(lambda x: x.lower() if isinstance(x, str) else x)

# 移除URL
url_pattern = re.compile(r'https?://\S+')
df['clean_posts'] = df['posts'].apply(lambda x: url_pattern.sub('', x))

# 移除标点符号和特殊字符
df = df.replace(to_replace=r'[^\w\s]', value='', regex=True)

# 移除数字
df = df.replace(to_replace=r'\d', value='', regex=True)

# 词汇化
df['clean_posts'] = df['clean_posts'].apply(word_tokenize)

# 移除停用词
stop_words = set(stopwords.words('english'))
df['clean_posts'] = df['clean_posts'].apply(lambda x: [word for word in x if word not in stop_words])

# 词干化
stemmer = PorterStemmer()
df['stemmed_clean_posts'] = df['clean_posts'].apply(lambda x: [stemmer.stem(word) for word in x])

# 词形还原
lemmatizer = WordNetLemmatizer()
def lemmatize_tokens(tokens):
    def get_wordnet_pos(word):
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
        return tag_dict.get(tag, wordnet.NOUN)
    return [lemmatizer.lemmatize(token, get_wordnet_pos(token)) for token in tokens]

df['lemmatized_clean_posts'] = df['clean_posts'].apply(lemmatize_tokens)

# 特征提取
X = df['lemmatized_clean_posts'].apply(lambda x: ' '.join(x))
y = df['type']

vect = CountVectorizer(ngram_range=(1, 3), min_df=5, max_features=8000)
X_vec = vect.fit_transform(X).toarray()

# 拆分数据集
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=0)

# 创建和训练 KNN 模型
knn = KNeighborsClassifier(n_neighbors=19)
knn.fit(X_train, y_train)

# 保存 CountVectorizer 和 KNN 模型
joblib.dump(vect, 'count_vectorizer.pkl')
joblib.dump(knn, 'knn_model-2.pkl')

print("特征提取器和模型已成功保存。")

# 测试模型准确性
y_pred_knn = knn.predict(X_test)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print("KNN 模型准确率:", accuracy_knn)

# 生成分类报告
classification_report_knn = classification_report(y_test, y_pred_knn, zero_division=0)
print("KNN 分类报告:")
print(classification_report_knn)
