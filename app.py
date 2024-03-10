import pandas as pd
import warnings
import json
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sn
from pandas.plotting import scatter_matrix
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import HashingVectorizer
import re
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

from flask import Flask, request
app = Flask(__name__)

@app.route('/', methods=['POST'])
def home():
   input_keywords= request.json['keyword']

   perfume = pd.read_csv(r'C:\Users\moons\Desktop\merge\Perfume_url.csv')
   perfume = perfume.loc[:, ~perfume.columns.str.contains('^Unnamed')]
   perfume.columns=['Id','Brand','Category','Name','Rating','Gender','Scents','Notes','Perfume url']

   #Note열 공백과 and 제거 후 새로운 df로 생성 
   new_df = pd.DataFrame(perfume.Notes.str.split(',| and| And ').tolist(), index=perfume.Name).stack()
   #다중 인덱스 삭제
   new_df = new_df.reset_index(level=1, drop=True).reset_index()
   new_df.columns = ['Name', 'Notes']
   #대소문자 구분X
   new_df['Notes'] = new_df['Notes'].str.strip().str.lower() 
   #선행,후행 공백 제거
   new_df['Notes'] = new_df['Notes'].str.strip()
   #Name열로 그룹화
   new_df = new_df.groupby('Name')['Notes'].apply(set).reset_index()
   #Note열을 쉼표로 구분된 문자열로 변환
   new_df['Notes'] = new_df['Notes'].apply(','.join)
   #이름열 기준으로 중복 행 제거
   new_df = new_df.drop_duplicates(subset='Name')
   #new_df에 Id, Name, Rating, Brand, Gender, Scents, Perfume url을 Name열을 기준으로 병합
   new_df = new_df.merge(perfume[['Id','Name', 'Rating','Brand','Gender','Scents','Perfume url']], on='Name', how='left')
   #Name이 첫번째 행으로 선택
   new_df = new_df.groupby('Name').first().reset_index()
   #'Id','Name', 'Brand','Gender','Rating', 'Scents','Notes', 'Perfume url'으로 정렬
   new_df = new_df[['Id','Name', 'Brand','Gender','Rating', 'Scents','Notes', 'Perfume url']]

   # hashing vectorizer with 2^10 features
   vectorizer = HashingVectorizer(n_features=641, binary=True)
   vector_size=641
   perfume_vectors = np.empty((0, vector_size))
   
   # 각 향수마다 해당하는 노트가 있는지를 나타내는 이진 벡터 만들기
   for i, p in enumerate(new_df.Name):
    notes_str = new_df[new_df.Name == p].Notes.values[0]
    notes_list = re.split(',| and| And ', notes_str)
    notes_hashed = vectorizer.transform(notes_list).toarray()
    vector = np.sum(notes_hashed, axis=0)
    perfume_vectors = np.concatenate((perfume_vectors, np.expand_dims(vector, axis=0)), axis=0)
    
    
# 입력받은 키워드에 대한 이진 벡터를 생성
   input_notes = re.split(',| and| And ', input_keywords,flags=re.IGNORECASE)
   input_vector = vectorizer.transform(input_notes).toarray()
   input_vector = np.sum(input_vector, axis=0).reshape(1, -1)
    
    
   # 코사인 유사도를 계산
   cosine_similarities = cosine_similarity(input_vector, perfume_vectors)
   similarity_scores = list(enumerate(cosine_similarities[0]))
   sorted_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    
    
    
   # 유사도가 임계값보다 높은 향수들만 추천
   threshold = 0.8
    
   top_perfumes = [(new_df['Name'][i], new_df['Brand'][i],new_df['Gender'][i],new_df['Rating'][i],new_df['Notes'][i]) for i, score in sorted_scores if score >= threshold]
    
   perfume_info = ""
   for name, brand, gender, rating, notes in top_perfumes:
      perfume_info += "Name: {}\n".format(name)
      perfume_info += "Brand: {}\n".format(brand)
      perfume_info += "Gender: {}\n".format(gender)
      perfume_info += "Rating: {}\n".format(rating)
      perfume_info += "Notes: {}\n".format(notes)
      perfume_info += "\n"

   return perfume_info

   
if __name__ == '__main__':  
   #app.run('0.0.0.0',port=5000,debug=True)
   app.run(host="0.0.0.0", port=5000, threaded=True)



