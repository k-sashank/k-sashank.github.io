#from fastapi import FastAPI
from flask import Flask, request
import json
import numpy as np
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import tokenizer_from_json

#app = FastAPI()
app = Flask(__name__)

@app.get('/')
def get_root():
    return {'message': 'This is the sentiment analysis app'}

@app.get('/sentiment_analysis/')
def query_sentiment_analysis():
    text = request.args.get('text')
    return analyze_sentiment(text)

def analyze_sentiment(text):
    sentence = []
    sentence.append(text)
    with open('tokenizer.json') as f:
        data = json.load(f)
        tokenizer = tokenizer_from_json(data)
    sequences = tokenizer.texts_to_sequences(sentence)
    padded = pad_sequences(sequences, maxlen=100, padding='post', truncating='post')
    model = load_model('text_analyzer.h5')
    result = model.predict(padded)
    print(result)
    result = result[0]
    print(result)

    max_index = np.argmax(result, axis=0)
    sentiment = ''

    if (max_index == 0):
        sentiment = 'Angry'
    elif (max_index == 1):
        sentiment = 'Happy'
    elif (max_index == 2):
        sentiment = 'Sad'

    prob = result[max_index]*100
    prob = round(prob, 2)

    return {'sentiment': sentiment, 'probability': prob}

app.run()
