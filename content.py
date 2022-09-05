import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv('articles.csv')
df = df[df['title_work'].notna()]

count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(df['title_work'])

cosine_sim2 = cosine_similarity(count_matrix, count_matrix)

indices = pd.Series(df.index, index=df['title_work'])

def get_recommendations(title):
    if isinstance(title, str):
        title = str.lower(title.replace(" ", ""))
    else:
        title = ''
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim2[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    article_indices = [i[0] for i in sim_scores]
    return df[["url", "title", "text", "lang", "total_events"]].iloc[article_indices].values.tolist()