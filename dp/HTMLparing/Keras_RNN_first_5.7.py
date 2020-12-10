import sys
import os,shutil
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, Dropout, TimeDistributed, Flatten
from keras.layers import LSTM
import numpy as np
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D




def fillComments(contents, i):
    sent5=[]#this is for return
    
    
    item1=""#2 before sentence
    item2=""#2 behind sentence
    item3=""#1 before sentence
    item4=""#1 behind sentence
    item5=""#current sentence
    
    #########current sentences
    item5=contents[i]
    
    
    #########fill before sentences
    lineBefore=3
    
    
    for kk in range(1,3):#[1,2] check where is the comment break, record as lineBefore
        if "####comment#12345#" in contents[i-kk] or "#####title#12345#" in contents[i-kk]:
           lineBefore=kk
           break
    
    if lineBefore==3:
        item1=contents[i-2]
        item3=contents[i-1]
        

    if lineBefore==2:
        item1="####comment#12345#"
        item3=contents[i-1]
        
        
    if lineBefore==1:
        item1="####comment#12345#"
        item3="####comment#12345#"
        
           
    ##########fill behind sentences
    lineBehind=3
    
    for kk in range(1,3):#[1,2] check where is the comment break, record as lineBefore
        if "####comment#12345#" in contents[i+kk] or "#####title#12345#" in contents[i+kk]:
           lineBehind=kk
           break
    
    
    if lineBehind==3:
        item2=contents[i+2]
        item4=contents[i+1]

    
    if lineBehind==2:
        item2="####comment#12345#"
        item4=contents[i+1]
    
    if lineBehind==1:
        item2="####comment#12345#"
        item4="####comment#12345#"
            
    
    sent5.append(item1)
    sent5.append(item2)
    sent5.append(item3)
    sent5.append(item4)
    sent5.append(item5)
    
    return sent5


def fillLabels(currentLine, allTrainY, allTrainZ):
    #########this is for the #step and #wtep label####
    if "#step" in currentLine or "#wtep#" in currentLine:
        allTrainY.append(1)
    else:
        allTrainY.append(0)
        
    #########this is for #oracle label############
    if "#oracle" in currentLine:
        allTrainZ.append(1)
    else:
        allTrainZ.append(0)



def extractAllTrain(allTrainX,allTrainY,allTrainZ):
    for fileName in pathDir:
        #f = open("dataSet/labeled/"+fileName)
        f = open(trainFolder+"/"+fileName)
        contents = f.readlines()
        
        titleTag=False
        commentTag=0
        
        for i in range(0,len(contents)-1):
            line=contents[i]
            
            sent5=[]
            if "#####title#12345#" in line:
                if i==0:
                    titleTag=True#this is the title
                    continue
                else:
                    titleTag=False
                    commentTag=1#it means the first sentence of a somment
                    continue
            
            if titleTag==True:#this is for the title
                sent5.append("#####title#12345#")
                sent5.append("#####title#12345#")
                sent5.append("#####title#12345#")
                sent5.append("#####title#12345#")
                sent5.append(contents[i])
                
                fillLabels(contents[i],allTrainY,allTrainZ)#fill labels for title
                
            elif commentTag>0:#this is for the comments
                if "####comment#12345#" in line:
                    commentTag=1
                    continue
                else:
                    sent5=fillComments(contents,i)
                    commentTag=commentTag+1
                    
                fillLabels(contents[i],allTrainY,allTrainZ)#fill labels for comments 
                sent5=fillComments(contents,i)
            allTrainX.append(sent5)
    
def VectcountTrans(allTrainX, vector):
    extendVector=[]
    
    for item in allTrainX:
        extendVector.append(item[0])
        extendVector.append(item[1])
        extendVector.append(item[2])
        extendVector.append(item[3])
        extendVector.append(item[4])
    
    countX=vector.fit_transform(extendVector).toarray()#len(count[0])=21041
    
    allTrainX=[]
    for i in range(0,len(countX)/5):
        sent5=[]
        '''
        sent5=np.append(sent5, countX[5*i], axis = 1)
        sent5=np.append(sent5, countX[5*i+1], axis = 1)
        sent5=np.append(sent5, countX[5*i+2], axis = 0)
        sent5=np.append(sent5, countX[5*i+3], axis = 0)
        sent5=np.append(sent5, countX[5*i+4], axis = 0)
        
        '''
        sent5.append(countX[5*i])
        sent5.append(countX[5*i+1])
        sent5.append(countX[5*i+2])
        sent5.append(countX[5*i+3])
        sent5.append(countX[5*i+4])
        allTrainX.append(sent5)
        
    
    
    return allTrainX
    
    

######main function

if __name__ == '__main__':

    global nlp
    nlp = spacy.load("en_core_web_sm")


    ##########train############
    targetFolder='google code'
    
    
    trainFolder='buildDataSet/'+targetFolder+'/train'
    testFolder='buildDataSet/'+targetFolder+'/test'
    
    
    
    
    
    
    pathDir=os.listdir(trainFolder)
    allTrainX=[]#sentence
    allTrainY=[]#step list
    allTrainZ=[]#crash list
    
    extractAllTrain(allTrainX,allTrainY,allTrainZ)


    ############countVector#########
    vector = CountVectorizer(ngram_range=(1,1))
    allTrainX=VectcountTrans(allTrainX, vector)#it is because the sklearn API can not handle the 3 dimensions vector.
    
    
    ###########we need to change the data format
    allTrainX=np.array(allTrainX)#  memery will explode, if we use this one.
    allTrainY=np.array(allTrainY)
    
    
    
    
    model = Sequential()
    #model.add(LSTM(100, batch_input_shape=(5, 500))) #(None, 100)
    model.add(LSTM(100, input_shape=(5,100))) #(None, 100)    
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(allTrainX, allTrainY, epochs=1, batch_size=64)
    model.summary()
    
    
    
    
    #vector = CountVectorizer(ngram_range=(1,1))#ngram_range=(1,2)#gram 1 and gram 2# the gram 3 will be very costly
    
    
    
    
    
    #x=vector.fit_transform(allTrainX).toarray()
    
    
    


            
            
    print("asd")
            
            


            
            
            
            
            
    

    

