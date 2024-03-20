import pickle
import gensim
import numpy as np
import json


caption_file = "datasets/DeepFashion/captions.json"

with open(caption_file, 'r') as f:
    captions = json.load(f)

total_doc = []
for key in captions:
    text = gensim.utils.simple_preprocess(captions[key])
    total_doc.append(text)

model = gensim.models.Word2Vec(
    window=10,
    min_count=2,
    workers=4
)

model.build_vocab(total_doc)
model.train(total_doc, total_examples=model.corpus_count, epochs=20)


with open('models/Inception/text_embedding_df.pkl', 'wb') as file:
    pickle.dump(model, file)


def get_query_vector1(query, model):
    new_query = gensim.utils.simple_preprocess(query)
    query_vector = sum(model.wv[word] for word in new_query) / len(new_query)
    return query_vector


def get_query_vector2(query, model):
    new_query = gensim.utils.simple_preprocess(query)

    query_vector = np.zeros(model.vector_size)
    for word in new_query:
        vec = np.array(model.wv[word])
        vec = vec/np.linalg.norm(vec)
        query_vector += vec

    query_vector = query_vector / len(new_query)
    return query_vector


similar_words = model.wv.most_similar(
    positive=[get_query_vector2("sleeveless shirt hat", model)], topn=10)
print(similar_words)
