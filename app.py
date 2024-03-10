#flask 연동
from flask import Flask
app = Flask(__name__)

@app.route('/')
def home():
   return '서버연결'

if __name__ == '__main__':  
   app.run('0.0.0.0',port=5000,debug=True)