#!/usr/bin/env python
# coding: utf-8

# In[2]:


#pip install tensorflow
#pip install keras
#pip install gensim
import pandas as pd
import numpy as np
#conda install numpy


df1 = pd.read_csv('C:\\Users\\USER\\Documents\\random_positive.csv')
df1.head(5)
df2=pd.read_csv('C:\\Users\\USER\\Documents\\random_negative.csv')
df2.head(5)
df3=pd.concat([df1,df2])
df3.shape
# 한번만 시행하고 재시행하지 말것

df3['nlabel']=np.where(df3['label']=='Positive',1,0)
df=df3.drop(['label'], axis=1)
df.head()

from keras.preprocessing.text import Tokenizer
t = Tokenizer()
t.fit_on_texts(df['review'])
vocab_size = len(t.word_index) + 1

X_encoded = t.texts_to_sequences(df['review'])
max_length=max(len(l) for l in X_encoded)

from keras.preprocessing.sequence import pad_sequences
X_train=pad_sequences(X_encoded, maxlen=max_length, padding='post')
print(X_train)


import numpy as np
embedding_dict = dict()
f = open('C:\\Users\\USER\\Documents\\GH\\내부플젝\\glove.6B\\glove.6B.100d.txt', encoding="utf8")
.
for line in f:
    word_vector = line.split()
    word = word_vector[0]
    word_vector_arr = np.asarray(word_vector[1:], dtype='float32') # 100개의 값을 가지는 array로 변환
    embedding_dict[word] = word_vector_arr
f.close()
print('%s개의 Embedding vector가 있습니다.' % len(embedding_dict))


print(embedding_dict['respectable'])
print(len(embedding_dict['respectable']))

embedding_matrix = np.zeros((vocab_size, 100))
# 단어 집합 크기의 행과 100개의 열을 가지는 행렬 생성. 값은 전부 0으로 채워진다.
np.shape(embedding_matrix)


print(t.word_index.items()) #보면은 markets market 도의어로 처리 안되어있음


for word, i in t.word_index.items(): # 훈련 데이터의 단어 집합에서 단어를 1개씩 꺼내온다.
    temp = embedding_dict.get(word) # 단어(key) 해당되는 임베딩 벡터의 100개의 값(value)를 임시 변수에 저장
    if temp is not None:
        embedding_matrix[i] = temp # 임수 변수의 값을 단어와 맵핑되는 인덱스의 행에 삽입


from keras.models import Sequential
from keras.layers import Dense, Embedding, Flatten

model = Sequential()
embedding_layer = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=max_length, trainable=False)


max_words=max_length


from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import re


max_fatures = 30000
tokenizer = Tokenizer(nb_words=max_fatures, split=' ')
tokenizer.fit_on_texts(df['review'].values)
X1 = tokenizer.texts_to_sequences(df['review'].values)
X1 = pad_sequences(X1)
Y1 = pd.get_dummies(df['nlabel']).values
X1_train, X1_test, Y1_train, Y1_test = train_test_split(X1,Y1, random_state = 42)
print(X1_train.shape,Y1_train.shape)
print(X1_test.shape,Y1_test.shape)

embed_dim = 100
lstm_out = 200
model = Sequential()
model.add(embedding_layer)
model.add(LSTM(128, dropout=0.5, return_sequences=False
               , input_shape=(max_length,embed_dim)))
model.add(Dense(2,activation='softmax')) #예측하고자 하는 수에 따라 숫자
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())

batch_size = 32
model.fit(X1_train, Y1_train, nb_epoch = 3, batch_size=batch_size, verbose = 2)

score,acc = model.evaluate(X1_test, Y1_test, verbose = 2, batch_size = batch_size)
print("score: %.2f" % (score))
print("acc: %.2f" % (acc))


pos_cnt, neg_cnt, pos_correct, neg_correct = 0, 0, 0, 0
for x in range(len(X1_test)):

    result = model.predict(X1_test[x].reshape(1,X1_test.shape[1]),batch_size=1,verbose = 2)[0]

    if np.argmax(result) == np.argmax(Y1_test[x]):
        if np.argmax(Y1_test[x]) == 0:
            neg_correct += 1
        else:
            pos_correct += 1

    if np.argmax(Y1_test[x]) == 0:
        neg_cnt += 1
    else:
        pos_cnt += 1



print("pos_acc", pos_correct/pos_cnt*100, "%")
print("neg_acc", neg_correct/neg_cnt*100, "%")

df4= pd.read_csv("C:\\Users\\USER\\Documents\\GH\\내부플젝\\data_vader_labelled_without stemLem\\cheong_label.csv")
df4.head()
df5= pd.read_csv("C:\\Users\\USER\\Documents\\GH\\내부플젝\\data_vader_labelled_without stemLem\\gung_label.csv")
df5.head()
df6= pd.read_csv("C:\\Users\\USER\\Documents\\GH\\내부플젝\\data_vader_labelled_without stemLem\\insadong_label.csv")
df6.head()
df7= pd.read_csv("C:\\Users\\USER\\Documents\\GH\\내부플젝\\data_vader_labelled_without stemLem\\chon_label.csv")
df7.head()


df4['lstm_label']=np.zeros(len(df4))
df4.review=df4.review.astype(str)
#vectorizing the tweet by the pre-fitted tokenizer instance
twt8 = tokenizer.texts_to_sequences(df4['review'])
#padding the tweet to have exactly the same shape as `embedding_2` input
#pad_sequences(X_encoded, maxlen=max_length, padding='post'
twt8 = pad_sequences(twt8, maxlen=max_length, dtype='int32', value=0)
print(twt8)



criteria = [df4['rating'].between(1, 2), df4['rating'].between(3, 5)]
values = [0, 1]

df4['rrating'] = np.select(criteria, values, 0)
df4.head()


sentiment= model.predict(twt8,batch_size=32,verbose = 2)
for i in range(len(twt8)):
    df4.iloc[i,7]=np.argmax(sentiment[i])


df4['lstm_label']=np.where(df4.lstm_label==1.0,1,0)
df4['nlstm_label']=np.where(df4.label=='Positive',1,0)
df4.head()

np.mean(df4.lstm_label == df4.rrating)

np.mean(df4.lstm_label==df4.nlstm_label)


df5['lstm_label']=np.zeros(len(df5))
df5.review=df5.review.astype(str)
twt2 = tokenizer.texts_to_sequences(df5['review'])

twt2 = pad_sequences(twt2, maxlen=max_length, dtype='int32', value=0)
print(twt2)



criteria = [df5['rating'].between(1, 2), df5['rating'].between(3, 5)]
values = [0, 1]

df5['rrating'] = np.select(criteria, values, 0)
df5.head()


sentiment= model.predict(twt2,batch_size=32,verbose = 2)
for i in range(len(twt2)):
    df5.iloc[i,7]=np.argmax(sentiment[i])



df5['lstm_label']=np.where(df5.lstm_label==1.0,1,0)
df5['nlstm_label']=np.where(df5.label=='Positive',1,0)
df5.head()



np.mean(df5.lstm_label == df5.rrating)

np.mean(df5.lstm_label == df5.nlstm_label)


df6['lstm_label']=np.zeros(len(df6))
df6.review=df6.review.astype(str)
twt3 = tokenizer.texts_to_sequences(df6['review'])

twt3 = pad_sequences(twt3, maxlen=max_length, dtype='int32', value=0)
print(twt3)


criteria = [df6['rating'].between(1, 2), df6['rating'].between(3, 5)]
values = [0, 1]

df6['rrating'] = np.select(criteria, values, 0)
df6.head()

sentiment= model.predict(twt3,batch_size=32,verbose = 2)
for i in range(len(twt3)):
    df6.iloc[i,7]=np.argmax(sentiment[i])

df6['lstm_label']=np.where(df6.lstm_label==1.0,1,0)
df6['nlstm_label']=np.where(df6.label=='Positive',1,0)
df6.head()


np.mean(df6.lstm_label == df6.nlstm_label)
n(df6.lstm_label == df6.rrating)


df7['lstm_label']=np.zeros(len(df7))
df7.review=df7.review.astype(str)

twt4 = tokenizer.texts_to_sequences(df7['review'])

twt4 = pad_sequences(twt4, maxlen=max_length, dtype='int32', value=0)
print(twt4)


criteria = [df7['rating'].between(1, 2), df7['rating'].between(3, 5)]
values = [0, 1]

df7['rrating'] = np.select(criteria, values, 0)
df7.head()


sentiment= model.predict(twt4,batch_size=32,verbose = 2)


for i in range(len(twt4)):
    df7.iloc[i,7]=np.argmax(sentiment[i])
df7.head()



df7['lstm_label']=np.where(df7['lstm_label']==1.0,1,0)
df7.head()

df7['nlstm_label']=np.where(df7.label=='Positive',1,0)
df7.head()


np.mean(df7.lstm_label == df7.nlstm_label)


np.mean(df7.lstm_label == df7.rrating)


ldf4= pd.read_csv("C:\\Users\\USER\\Documents\\cheong_lemmatized.csv")
ldf4.head()


ldf4['lstm_label']=np.zeros(len(ldf4))
ldf4.review=ldf4.review.astype(str)

ltwt8 = tokenizer.texts_to_sequences(ldf4['review'])

ltwt8 = pad_sequences(ltwt8, maxlen=max_length, dtype='int32', value=0)
print(ltwt8)


criteria = [ldf4['rating'].between(1, 2), ldf4['rating'].between(3, 5)]
values = [0, 1]

ldf4['rrating'] = np.select(criteria, values, 0)
ldf4.head()


sentiment= model.predict(ltwt8,batch_size=32,verbose = 2)
for i in range(len(ltwt8)):
    ldf4.iloc[i,3]=np.argmax(sentiment[i])


ldf4['lstm_label']=np.where(ldf4.lstm_label==1.0,1,0)
ldf4['nlstm_label']=np.where(ldf4.label=='Positive',1,0)
ldf4.head()

np.mean(ldf4.lstm_label == ldf4.nlstm_label)

np.mean(ldf4.lstm_label == ldf4.rrating)


ldf4.to_csv('C:\\Users\\USER\\Documents\\relemmatized_cheong.csv',index=False)

ldf5= pd.read_csv("C:\\Users\\USER\\Documents\\gung_lemmatized.csv")
ldf5.head()

ldf5['lstm_label']=np.zeros(len(ldf5))
ldf5.review=ldf5.review.astype(str)

ltwt5 = tokenizer.texts_to_sequences(ldf5['review'])

ltwt5 = pad_sequences(ltwt5, maxlen=max_length, dtype='int32', value=0)
print(ltwt5)

criteria = [ldf5['rating'].between(1, 2), ldf5['rating'].between(3, 5)]
values = [0, 1]

ldf5['rrating'] = np.select(criteria, values, 0)
ldf5.head()

sentiment= model.predict(ltwt5,batch_size=32,verbose = 2)
for i in range(len(ltwt5)):
    ldf5.iloc[i,3]=np.argmax(sentiment[i])

ldf5['lstm_label']=np.where(ldf5.lstm_label==1.0,1,0)
ldf5['nlstm_label']=np.where(ldf5.label=='Positive',1,0)
ldf5.head()




np.mean(ldf5.lstm_label == ldf5.nlstm_label)
np.mean(ldf5.lstm_label == ldf5.rrating)
ldf5.to_csv('C:\\Users\\USER\\Documents\\relemmatized_gung.csv',index=False)


ldf6= pd.read_csv("C:\\Users\\USER\\Documents\\insadong_lemmatized.csv")
ldf6.head()

ldf6['lstm_label']=np.zeros(len(ldf6))
ldf6.review=ldf6.review.astype(str)

ltwt6 = tokenizer.texts_to_sequences(ldf6['review'])

ltwt6 = pad_sequences(ltwt6, maxlen=max_length, dtype='int32', value=0)
print(ltwt6)

criteria = [ldf6['rating'].between(1, 2), ldf6['rating'].between(3, 5)]
values = [0, 1]

ldf6['rrating'] = np.select(criteria, values, 0)
ldf6.head()

sentiment= model.predict(ltwt6,batch_size=32,verbose = 2)
for i in range(len(ltwt6)):
    ldf6.iloc[i,3]=np.argmax(sentiment[i])

ldf6['lstm_label']=np.where(ldf6.lstm_label==1.0,1,0)
ldf6['nlstm_label']=np.where(ldf6.label=='Positive',1,0)
ldf6.head()


np.mean(ldf6.lstm_label == ldf6.nlstm_label)
np.mean(ldf6.lstm_label == ldf6.rrating)
ldf6.to_csv('C:\\Users\\USER\\Documents\\relemmatized_insadong.csv',index=False)



ldf7= pd.read_csv("C:\\Users\\USER\\Documents\\chon_lemmatized.csv")
ldf7.head()

ldf7['lstm_label']=np.zeros(len(ldf7))
ldf7.review=ldf7.review.astype(str)

ltwt7 = tokenizer.texts_to_sequences(ldf7['review'])

ltwt7 = pad_sequences(ltwt7, maxlen=max_length, dtype='int32', value=0)
print(ltwt7)

criteria = [ldf7['rating'].between(1, 2), ldf7['rating'].between(3, 5)]
values = [0, 1]

ldf7['rrating'] = np.select(criteria, values, 0)
ldf7.head()

sentiment= model.predict(ltwt7,batch_size=32,verbose = 2)
for i in range(len(ltwt7)):
    ldf7.iloc[i,3]=np.argmax(sentiment[i])

ldf7['lstm_label']=np.where(ldf7.lstm_label==1.0,1,0)
ldf7['nlstm_label']=np.where(ldf7.label=='Positive',1,0)
ldf7.head()


ldf77=ldf7.drop(['rating', 'label'], axis=1)
ldf77.head(15)



np.mean(ldf7.lstm_label == ldf7.nlstm_label)
np.mean(ldf7.lstm_label == ldf7.rrating)
ldf7.to_csv('C:\\Users\\USER\\Documents\\relemmatized_chon.csv',index=False)


df7.to_csv('C:\\Users\\USER\\Documents\\re_bukchon.csv',index=False)



df6.to_csv('C:\\Users\\USER\\Documents\\re_insadong.csv',index=False)


df4.to_csv('C:\\Users\\USER\\Documents\\re_cheong.csv',index=False)


df5.to_csv('C:\\Users\\USER\\Documents\\re_gung.csv',index=False)


# In[ ]:


#p=1,n=0
#lstm_label: lstm 모델 돌려서 나온 결과
#rrating: rate 3-5 는 positivie,1   , rate1-2 는 negative,0
#nlstm_label: vader 기준으로 나눈 p, n
