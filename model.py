import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import re

# 下载必要的NLTK数据
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# 数据预处理函数
def preprocess_data(df):
    df = df.applymap(lambda x: x.lower() if isinstance(x, str) else x)
    
    # 移除URL和HTML标签
    url_pattern = re.compile(r'https?://\S+')
    df['clean_posts'] = df['posts'].apply(lambda x: url_pattern.sub('', x))
    df['clean_posts'] = df['clean_posts'].replace(to_replace=r'<.*?>', value='', regex=True)
    df['clean_posts'] = df['clean_posts'].replace(to_replace=r'[^\w\s]', value='', regex=True)
    df['clean_posts'] = df['clean_posts'].replace(to_replace=r'\d', value='', regex=True)
    
    # 分词
    tokenizer = RegexpTokenizer(r'\w+')
    df['clean_posts'] = df['clean_posts'].apply(tokenizer.tokenize)
    
    # 移除停用词
    stop_words = set(stopwords.words('english'))
    custom_stop_words = {'example', 'another'}  # 添加自定义停用词
    stop_words = stop_words.union(custom_stop_words)
    df['clean_posts'] = df['clean_posts'].apply(lambda x: [word for word in x if word not in stop_words])
    
    # 词形还原
    lemmatizer = WordNetLemmatizer()
    def lemmatize_tokens(tokens):
        def get_wordnet_pos(word):
            tag = nltk.pos_tag([word])[0][1][0].upper()
            tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
            return tag_dict.get(tag, wordnet.NOUN)
        return [lemmatizer.lemmatize(token, get_wordnet_pos(token)) for token in tokens]

    df['lemmatized_clean_posts'] = df['clean_posts'].apply(lemmatize_tokens)
    
    return df

# 加载和预处理数据
df = pd.read_csv('mbti_1.csv')
df = preprocess_data(df)

# 使用TF-IDF进行特征提取
X = df['lemmatized_clean_posts'].apply(lambda x: ' '.join(x))
y = df['type']

tfidf_vect = TfidfVectorizer(ngram_range=(1, 3), max_features=10000, smooth_idf=True, norm='l2')
X_vec = tfidf_vect.fit_transform(X).toarray()

# 拆分数据集
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=0)

# KNN的超参数调优
param_grid = {'n_neighbors': [3, 5, 7, 9, 11, 15, 19]}
knn = KNeighborsClassifier()
grid_search = GridSearchCV(knn, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# 最佳KNN模型
best_knn = grid_search.best_estimator_

# 随机森林模型
rf_model = RandomForestClassifier(n_estimators=100, random_state=0)
rf_model.fit(X_train, y_train)

# 支持向量机模型
svm_model = SVC(kernel='linear', random_state=0)
svm_model.fit(X_train, y_train)

# 保存向量化器和模型
joblib.dump(tfidf_vect, 'tfidf_vectorizer.pkl')
joblib.dump(best_knn, 'knn_model.pkl')
joblib.dump(rf_model, 'rf_model.pkl')
joblib.dump(svm_model, 'svm_model.pkl')

print("特征提取器和模型已成功保存。")

# 测试模型准确性
for model, name in zip([best_knn, rf_model, svm_model], ['KNN', 'Random Forest', 'SVM']):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} 模型准确率: {accuracy}")
    print(f"{name} 分类报告:\n", classification_report(y_test, y_pred))