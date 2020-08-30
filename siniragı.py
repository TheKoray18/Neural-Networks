#%% Veriyi Çağırma
import numpy as np
import pandas as pd

data=pd.read_csv("Churn_Modelling.csv")

#Verimize Bakıyoruz

info=data.info()
print(info)

isnull=data.isnull()
print(isnull)

#%% Veri Önişleme

#RowNumber,Customerld ve Surname Sütunlarını Cıkardık

exited=data.iloc[:,-1].values

data=data.iloc[:,3:13].values


#%% Georaphy ve Gender Label Encoder Yapıyoruz.Kategorik >> Numeric

from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

data[:,1]=le.fit_transform(data[:,1])

le2=LabelEncoder()

data[:,2]=le2.fit_transform(data[:,2])


#%% One Hot Encoder

#Column Transformer kullanıyoruz

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ohe=ColumnTransformer([("ohe", OneHotEncoder(dtype=float),[1])],remainder="passthrough")

data=ohe.fit_transform(data)

data=data[:,1:]

#%% Train ve Test Olarak Ayrılması

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(data,exited,test_size=0.33,random_state=0)

#%% Train ve Test verilerimizin boyutlarına bakalım

print("X_train:",x_train.shape)
print("x_test:",x_test.shape)
print("Y_train:",y_train.shape)
print("Y_test:",y_test.shape)

#%% X_train ve X_test verilerimizi ölçeklendiriyoruz

from sklearn.preprocessing import StandardScaler

ss=StandardScaler()

x_train=ss.fit_transform(x_train)

x_test=ss.fit_transform(x_test)

#%% Yapay Sinir Ağı Oluşturuyoruz
#Activasyon Fonks. olarak relu ve sigmoid kullanıyoruz

from keras.models import Sequential
from keras.layers import Dense

model=Sequential()

model.add(Dense(9, init='uniform', activation='relu',input_dim=11))
model.add(Dense(9, init='uniform', activation='relu'))
model.add(Dense(9, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))

#%% Compile Ediyoruz

model.compile(optimizer='Adam',loss="binary_crossentropy", metrics=["accuracy"])

#%% Modeli fit ediyoruz

model.fit(x_train,y_train,epochs=80)

#%% Predict

y_pred=model.predict(x_test)

#%% Sadece 1 veya 0 olsun istediğimiz için True veya False Boolean deger yapıyoruz

y_pred=(y_pred>0.5)

#%% Confusion Matrix olusturuyoruz

from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test, y_pred)

#%% Modelimizin Bilgilerini Öğreniyoruz

model.summary()

#%% Test datasının Loss ve Accuracy sonuçları

score=model.evaluate(x_test,y_test,verbose=1)

print("test_loss:",score[0])
print("test_accuracy:",score[1])










