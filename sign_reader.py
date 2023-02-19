import numpy as np 
import pandas as pd
import cv2
import tensorflow as tf
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout

#------------------------------------user input-------------------------------------
sizeOFtest=0.2 #test size

#image_loader=input("want to load previous np arrayes [y/n]?")
#image_loader=image_loader.lower()
image_loader='n'

#model_loader=input("want to load previous model [y/n]?")
#model_loader=model_loader.lower()
model_loader='n'

epochs = 15
#------------------------------------data input-------------------------------------
cur_path=cur_path = os.getcwd()
if image_loader=='y':   #loads previous np arrays that contains image values and labels
    data =np.load('data.npy')
    labels =np.load('labels.npy')
    extra_data=np.load('extra_data.npy')
    print('loading successful')

elif image_loader=='n': #reads images and assigns them to np arrays
    data = []
    labels = []
    path = os.path.join(cur_path,'data1')
    images = os.listdir(path)
    for pic in images:
        try:
            labels.append(int(pic[0:3]))
            image = Image.open(path + '\\'+ pic)
            image = image.resize((30,30))
            image = np.array(image)
            data.append(image)
        except:
            print("Error loading image")
    data = np.array(data)
    labels = np.array(labels)
    np.save('data.npy',data)
    np.save('labels.npy',labels)
    print("loading successful")

else:print('invalid input')


#------------------------------------data split-------------------------------------
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=sizeOFtest, random_state=0)
class_size=len(set(labels))
y_train = to_categorical(y_train, class_size)
y_test = to_categorical(y_test, class_size)
#------------------------------------model-------------------------------------
if model_loader=='y':   #load previous model
    model=load_model("brain.h5")
    
elif model_loader=='n': # makes a new model
    model = Sequential()
    
    #input layer
    model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=X_train.shape[1:]))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.25))
    
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.25))
    
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(rate=0.25))
    
    #output layer
    model.add(Dense(class_size, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history=model.fit(X_train, y_train, batch_size=32, epochs=epochs, validation_data=(X_test, y_test))
    model.save("brain.h5")

else: print('invalid command')

#------------------------------------evaluation-------------------------------------
results = model.evaluate(X_test,y_test,batch_size=32)
print("test Accuracy:" + str(round(results[1],4))+"    test loss: "+str(round(results[0],4)))

#------------------------------------extra-------------------------------------
def image_class(image_path): #takes an image path and prints class and probability of the answer being right
    test_data=[]
    model=load_model("brain.h5")
    image=Image.open(image_path)
    image = image.resize((30,30))
    image = np.array(image)
    test_data.append(image)
    test_data.append(image)
    test_data = np.array(test_data)
    d=model.predict(test_data)
    clas=d[0].argmax()
    probability=d[0][clas]
    print("the class is: "+str(clas)+"     confidence:"+ str(probability))

