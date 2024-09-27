import joblib
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import nltk

# 下载必要的NLTK数据
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# 加载保存的 TfidfVectorizer 和 KNN 模型
tfidf_vect = joblib.load('tfidf_vectorizer.pkl')
knn = joblib.load('knn_model.pkl')

# 数据预处理函数（与训练时相同）
def preprocess_new_post(post):
    # 移除URL
    url_pattern = re.compile(r'https?://\S+')
    post_clean = url_pattern.sub('', post)

    # 移除标点符号和特殊字符
    post_clean = re.sub(r'[^\w\s]', '', post_clean)

    # 移除数字
    post_clean = re.sub(r'\d', '', post_clean)

    # 词汇化
    tokens = word_tokenize(post_clean.lower())

    # 移除停用词
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # 词形还原
    lemmatizer = WordNetLemmatizer()
    def lemmatize_tokens(tokens):
        def get_wordnet_pos(word):
            tag = nltk.pos_tag([word])[0][1][0].upper()
            tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
            return tag_dict.get(tag, wordnet.NOUN)
        return [lemmatizer.lemmatize(token, get_wordnet_pos(token)) for token in tokens]
    
    lemmatized_tokens = lemmatize_tokens(tokens)
    return ' '.join(lemmatized_tokens)

# 定义新帖子以进行预测
new_post = "I like to spend time with my friends but I'm also happy when I'm by myself"

# 数据预处理
new_post_processed = preprocess_new_post(new_post)

# 转换为特征向量
X_new = tfidf_vect.transform([new_post_processed])

# 使用 KNN 模型进行预测
predicted_mbti = knn.predict(X_new)

print("Predicted MBTI Type:", predicted_mbti[0])
