from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt

import json
import numpy as np
import random

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
#by 이인규, 박도영 챗봇 train을 위한 python 소스 작성
#by 노민성,정종현 python 소스를 views.py 변환
def start(request):
    context = {}
    return render(request, "start.html",context)

def home(request):
    context = {}

    return render(request, "chathome.html", context)
# def Global(): #key_words전역 변수선언            예외처리할 keyword기능 구현중 2018038077박도영
#     global key_words
#     key_words=[]


#by edited 이인규/박도영 
#데이터 학습함수
@csrf_exempt
def chattrain(request):
    # Global() #전역변수호출 예외처리할 keyword기능 구현중 2018038077박도영
    context = {}

    print('chattrain ---> +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    
    #by 박도영 파일을 연 후 리스트에 데이터파일의 text 항목 넣기
    # file = open(f"{CURRENT_WORKING_DIRECTORY}intents.json", encoding="UTF-8")
    file = open(f"./static/intents.json", encoding="UTF-8")
    data = json.loads(file.read())

    # js = json.loads(data.decode("utf-8"))

    training_sentences = [] #트레이닝할 문장 리스트 형성 ,밑에 labels은 문장에 맞는 tag, responses는 문장에 대응되는 reponse들, image는 문장에 대응되는 image 유무리스트
    training_labels = [] #tag
    labels = [] #
    responses = []
    image=[]
    #keyword=[]  예외처리할 keyword기능 구현중 2018038077박도영
    
    #by 박도영 리스트에 데이터 넣기
    for intent in data['intents']:
        for pattern in intent['patterns']:
            training_sentences.append(pattern)
            words = pattern.split()
            for i in range(len(words) - 1): #문장의 단어집합을 형성해 챗봇대답의 정확성을 더 높임 //
                training_sentences.append(words[i] + " " + words[i + 1])
            training_labels.append(intent['tag'])
            for j in range(len(pattern.split()) - 1):
                training_labels.append(intent['tag'])       
        responses.append(intent['responses'])
        #keyword.append(intent['keywords'])  예외처리할 keyword기능 구현중 2018038077박도영
        #by 이인규 이미지 key 추가
        image.append(intent['image']) #이미지 유무

        if intent['tag'] not in labels:
            labels.append(intent['tag'])
    
    #by 출처에 tokenization/modeling 참고
    num_classes = len(labels) 

    lbl_encoder = LabelEncoder() #자연어를 인코딩 하는과정
    lbl_encoder.fit(training_labels)
    training_labels = lbl_encoder.transform(training_labels)

    vocab_size = 1000
    embedding_dim = 16
    max_len = 20
    oov_token = "<OOV>" #예외처리

    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token) #인코딩 한것을 토큰화
    tokenizer.fit_on_texts(training_sentences)
    word_index = tokenizer.word_index
    sequences = tokenizer.texts_to_sequences(training_sentences)
    padded_sequences = pad_sequences(sequences, truncating='post', maxlen=max_len)

    # Model Training
    #open source 사용 : https://github.com/amilavm/Chatbot_Keras
    model = Sequential() #트레이닝
    model.add(Embedding(vocab_size, embedding_dim, input_length=max_len))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(16, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
 
    model.summary()

    epochs = 500 #학습횟수 500회
    history = model.fit(padded_sequences, np.array(training_labels), epochs=epochs)

    # to save the trained model
    model.save("static/chat_model")

    import pickle

    import colorama
    colorama.init()
    from colorama import Fore, Style, Back
    
    # to save the fitted tokenizer
    with open('static/tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # to save the fitted label encoder
    with open('static/label_encoder.pickle', 'wb') as ecn_file:
        pickle.dump(lbl_encoder, ecn_file, protocol=pickle.HIGHEST_PROTOCOL)

    context['result_msg'] = 'Success'
    return JsonResponse(context, content_type="application/json")

#by 박도영/이인규
@csrf_exempt
def chatanswer(request):
    #by 이인규
    context = {}

    questext = request.GET['questext']

    import pickle

    import colorama
    colorama.init()
    from colorama import Fore, Style, Back

    file = open(f"./static/intents.json", encoding="UTF-8")
    data = json.loads(file.read())
    
    #by 박도영/이인규
    def chat3(inp):
        #by 박도영
        # load trained model
        model = keras.models.load_model('static/chat_model')

        # load tokenizer object
        with open('static/tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)

        # load label encoder object
        with open('static/label_encoder.pickle', 'rb') as enc:
            lbl_encoder = pickle.load(enc)

        # parameters
        max_len = 20

        # while True:
        print(Fore.LIGHTBLUE_EX + "User: " + Style.RESET_ALL, end="")
        # inp = 'What is name'
        
        result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([inp]),
                                                                          truncating='post', maxlen=max_len))
        #intents에서 tag에 맞는 값들 찾아내기
        tag = lbl_encoder.inverse_transform([np.argmax(result)])
        for intent in data['intents']:
            if intent['tag'] == tag:
                txt1 = np.random.choice(intent['responses']) 
                # for keyword in intent['keywords']:  예외처리할 keyword기능 구현중 2018038077박도영
                #     key_words.append(keyword)#key_words에 해당 keyword싹다 넣는다
                # for key_word in key_words: 
                #     while txt1.find(key_word)!=-1: #txt1에서 keyword를 발견했다면 ,keyword가 없는 reponse가 부여되어야지 빠져나간다
                #         txt1 = np.random.choice(intent['responses'])
                #by 이인규
                image1 = intent['image'] #image
                print(Fore.GREEN + "ChatBot:" + Style.RESET_ALL, txt1)

        # print(Fore.GREEN + "ChatBot:" + Style.RESET_ALL,random.choice(responses))

        return txt1,image1
    
    #by 이인규
    anstext = chat3(questext)[0]
    #image
    image=chat3(questext)[1]
    print(anstext)

    context['anstext'] = anstext
    context['flag'] = '0'
    #image
    context['image'] =image

    return JsonResponse(context, content_type="application/json")
