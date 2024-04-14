import re
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK resources (stopwords and punkt tokenizer)
# nltk.download('stopwords')
# nltk.download('punkt')


def preprocess_string(input_string):
    stopwords_list = set(stopwords.words('english'))
    content = input_string.lower()

    content = re.sub(r"\d", "", content)
    content = re.sub(r'[^\w\s]', '', content)

    words = word_tokenize(content)
    words = [w for w in words if w not in stopwords_list]

    return words


def get_query_vector2(query, model):
    new_query = preprocess_string(query)
    query_vector = np.zeros(model.vector_size)
    for word in new_query:
        if word in model.wv:
            vec = np.array(model.wv[word])
            vec = vec/np.linalg.norm(vec)
            query_vector += vec

    query_vector = query_vector / len(new_query)
    return query_vector
