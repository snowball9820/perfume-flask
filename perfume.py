import pandas as pd
import warnings
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sn
from pandas.plotting import scatter_matrix
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import HashingVectorizer
import re

warnings.filterwarnings('ignore')

perfume = pd.read_csv(r'C:\Users\moons\Desktop\merge\Perfume_url.csv')

perfume = perfume.loc[:, ~perfume.columns.str.contains('^Unnamed')]
perfume.columns = ['Id', 'Brand', 'Category', 'Name', 'Rating', 'Gender', 'Scents', 'Notes', 'Perfume url']

new_df = pd.DataFrame(perfume.Notes.str.split(',| and| And ').tolist(), index=perfume.Name).stack()
new_df = new_df.reset_index(level=1, drop=True)
new_df = new_df.reset_index()
new_df.columns = ['Name', 'Notes']
new_df['Notes'] = new_df['Notes'].str.strip()
new_df = new_df.groupby('Name')['Notes'].apply(set).reset_index()
new_df['Notes'] = new_df['Notes'].apply(','.join)



# 모든 향수들의 노트를 추출하여 중복을 제거
notes = list(set(new_df['Notes']))

# Create a hashing vectorizer with 2^10 features
vectorizer = HashingVectorizer(n_features=641, binary=True)

# 각 향수마다 해당하는 노트가 있는지를 나타내는 이진 벡터
perfume_vectors = []
for i, p in enumerate(new_df.Name):
    notes_str = new_df[new_df.Name == p].Notes.values[0]
    notes_list = re.split(',| and| And ', notes_str)
    notes_hashed = vectorizer.transform(notes_list).toarray()
    vector = np.sum(notes_hashed, axis=0)
    perfume_vectors.append(vector)
perfume_vectors = np.array(perfume_vectors)

# 입력받은 키워드에 대한 이진 벡터를 생성
input_keywords = input("키워드를 입력하세요: ")
input_notes = re.split(',| and| And ', input_keywords)
input_vector = vectorizer.transform(input_notes).toarray()
input_vector = np.sum(input_vector, axis=0).reshape(1, -1)

# 코사인 유사도를 계산
cosine_similarities = cosine_similarity(input_vector, perfume_vectors)
similarity_scores = list(enumerate(cosine_similarities[0]))
sorted_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)


# 유사도가 임계값보다 높은 향수들만 추천
threshold = 0.6

# 상위 6개 추천
top_perfumes = [perfume['Name'][i] for i, score in sorted_scores if score >= threshold][:6]

print("추천하는 향수는:", top_perfumes)
