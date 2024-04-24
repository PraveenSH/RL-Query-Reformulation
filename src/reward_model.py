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
        total = 0.0

        dups = len(sentence2[i].lower().split())
        uniques = len(words2)
        rep_reward = 0.0

        if dups > 0:
            rep_reward = uniques / dups

        for word in words2:

            if word in stop_words:
                continue

            if word in words1:
                continue

            total += 1.0
            word_emb = model.encode(word).reshape(1, -1)
            weight += (cosine_similarity(sent_emb, word_emb)[0][0] + 1) / 2

        alpha = 0.3
        if total == 0:
            rewards.append(rep_reward * alpha)
        else:
            rewards.append( (1-alpha) * (weight / total) + alpha * rep_reward)

    return rewards


if __name__ == "__main__":

    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    sentence1 = ["double door fridge"]
    sentence2 = ["double door fridge freezer double door fridge freezer double door fridge freezer double door fridge freezer double door fridge"]

    stop_words = set(stopwords.words('english'))
    print(similarity_reward(sentence1, sentence2, model), paraphrase_reward(sentence1, sentence2, stop_words, model))
