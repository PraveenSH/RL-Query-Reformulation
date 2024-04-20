import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import nltk


def similarity_reward(sentence1, sentence2, model):
    sentence_embeddings1 = model.encode(sentence1)
    sentence_embeddings2 = model.encode(sentence2)
    cosine_sim = cosine_similarity(sentence_embeddings1, sentence_embeddings2)
    normalized_similarity = (cosine_sim + 1) / 2
    similarities = []
    for i in range(len(sentence1)):
        similarities.append(normalized_similarity[i][i])
    return similarities


def paraphrase_reward(sentence1, sentence2, stop_words, model):

    rewards = []
    for i in range(len(sentence1)):
        sent_emb = model.encode(sentence1[i]).reshape(1, -1)
        words1 = set(sentence1[i].lower().split())
        words2 = set(sentence2[i].lower().split())

        weight = 0.0
        total = 0
        for word in words2:

            if word in stop_words:
                continue

            if word in words1:
                continue

            word_emb = model.encode(word).reshape(1, -1)
            total += 1
            weight += (cosine_similarity(sent_emb, word_emb)[0][0] + 1) / 2

        if total == 0:
            rewards.append(0.0)
        else:
            rewards.append(weight / total)

    return rewards


if __name__ == "__main__":

    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    sentence1 = ["bake a perfect cookie"]
    sentence2 = ["perfection cookies cookie baking great good cookie good cookies"]

    stop_words = set(stopwords.words('english'))
    print(similarity_reward(sentence1, sentence2, model), paraphrase_reward(sentence1, sentence2, stop_words, model))
