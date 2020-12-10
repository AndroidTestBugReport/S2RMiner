import sys
import os,shutil
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, Dropout, TimeDistributed, Flatten, Bidirectional
from keras.layers import TimeDistributed,Conv1D,Dense,Embedding,Input,Dropout,LSTM,Bidirectional,MaxPooling1D,Flatten,concatenate
import numpy as np
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import re
from gensim.test.utils import get_tmpfile
from sklearn.utils import class_weight
from imblearn.over_sampling import SMOTE
from keras.models import Model
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer




def show5sents(allTestsent_i, whyText):
    print("")
    print("")
    print("---"+whyText+"---")
    print("sent1:  "+allTestsent_i[0])
    print("sent2:  "+allTestsent_i[1])
    print("sent3:  "+allTestsent_i[2])
    print("sent4:  "+allTestsent_i[3])
    print("sent5_target:  "+allTestsent_i[4])
    print("")
    print("")
    print("")



def fillComments(contents, i, TestBol,sent5Text):
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
            
    
    #sent5.append(doc2VecTrans(item1))
    #sent5.append(doc2VecTrans(item2))
    #sent5.append(doc2VecTrans(item3))
    #sent5.append(doc2VecTrans(item4))
    sent5.append(doc2VecTrans(item5))
    
    
    if TestBol:
        #sent5Text.append(item1)
        #sent5Text.append(item2)
        #sent5Text.append(item3)
        #sent5Text.append(item4)
        sent5Text.append(item5)
    
    
    return sent5


def fillLabels(currentLine, allTrainY, allTrainZ):
    #########this is for the #step and #wtep label####
    if "#step" in currentLine or "#wtep" in currentLine:
        allTrainY.append(1)
    else:
        allTrainY.append(0)
        
    #########this is for #oracle label############
    if "#oracle" in currentLine:
        allTrainZ.append(1)
    else:
        allTrainZ.append(0)

def doc2VecTrans(oneLine):
    
    oneLine=oneLine.replace("#step","")
    oneLine=oneLine.replace("#wtep","")
    oneLine=oneLine.replace("#oracle","")
    oneLine=oneLine.strip()
            #wordList=oneLine.split(" ")###this one is with punctuation
    wordsSentList=re.findall(r"[\w']+|[.,!?;]", oneLine)####this one is without punctuation
    vector=modelDoc2vec.infer_vector(wordsSentList)
    
    return vector
    

def extractAllTrain(allTrainX,allTrainY,allTrainZ, pathDir, Folder, allTestsent, TestBol):#### TestBol and sent5Text are just to use check FP and FN
    
    #testCount=200######3this is just for test
    
    for fileName in pathDir:
        '''
        testCount=testCount-1
        if testCount==0:
            break
        '''
        
        
        #f = open("dataSet/labeled/"+fileName)
        f = open(Folder+"/"+fileName)
        contents = f.readlines()
        
        titleTag=False
        commentTag=0
        
        for i in range(0,len(contents)-1):
            line=contents[i]
            
            sent5=[]
            sent5Text=[]
            if "#####title#12345#" in line:
                if i==0:
                    titleTag=True#this is the title
                    continue
                else:
                    titleTag=False
                    commentTag=1#it means the first sentence of a somment
                    continue
            
            if titleTag==True:#this is for the title
                #sent5.append(doc2VecTrans("#####title#12345#"))
                #sent5.append(doc2VecTrans("#####title#12345#"))
                #sent5.append(doc2VecTrans("#####title#12345#"))
                #sent5.append(doc2VecTrans("#####title#12345#"))
                sent5.append(doc2VecTrans(contents[i]))
                
                fillLabels(contents[i],allTrainY,allTrainZ)#fill labels for title
                
                if TestBol:
                    #sent5Text.append("#####title#12345#")
                    #sent5Text.append("#####title#12345#")
                    #sent5Text.append("#####title#12345#")
                    #sent5Text.append("#####title#12345#")
                    sent5Text.append(contents[i])
                
                
                
                
            elif commentTag>0:#this is for the comments
                if "####comment#12345#" in line:
                    commentTag=1
                    continue
                else:
                    sent5=fillComments(contents,i,TestBol,sent5Text)#### TestBol and sent5Text are just to use check FP and FN
                    commentTag=commentTag+1
                    
                fillLabels(contents[i],allTrainY,allTrainZ)#fill labels for comments 
                #sent5=fillComments(contents,i,TestBol,sent5Text)  7.18 I do not know why do I add this before, so I remove it
            allTrainX.append(sent5)
            if TestBol:
                allTestsent.append(sent5Text[0].replace("#oracle","").replace("#step","").replace("#wtep",""))

######main function

if __name__ == '__main__':
    
    ##########train############
    targetFolder='google code'
    
    target="step"
    
    trainFolder='buildDataSet/'+targetFolder+'/train'
    testFolder='buildDataSet/'+targetFolder+'/test'
    
    fname="./my_doc2vec_model"
    modelDoc2vec = Doc2Vec.load(fname)
    
    
    
    trainPathDir=os.listdir(trainFolder)
    allTrainX=[]#sentence
    allTrainY=[]#step list
    allTrainZ=[]#crash list
    
    allTrainsent=[]
    TestBol=True
    
    extractAllTrain(allTrainX,allTrainY,allTrainZ,trainPathDir,trainFolder, allTrainsent, TestBol)
    
    if target=="oracle":
        allTrainY=allTrainZ
    
    
    
    #allTrainX=np.array(allTrainX)
    #allTrainY=np.array(allTrainY)
    
    print("afterPretrained")
    
    
    
    
    #allTrainX=allTrainX.reshape(-1,200)   useful for doc2vec
    
    
    ##########here is for smote#######################
    
    
    #allTrainX=allTrainX.reshape
    
    #allTrainX=allTrainX.reshape(-1,1000)
    
    #sm = SMOTE()
    #allTrainX, allTrainY = sm.fit_resample(allTrainX, allTrainY)
    
    #allTrainX=allTrainX.reshape(-1,5,200)
    
    
    
    ##################################################
    '''
    model = Sequential()
    #model.add(Bidirectional(LSTM(200)))
    model.add(Dense(200,activation='relu'))
    model.add(Dense(200,activation='relu'))
    model.add(Dense(200,activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['mse'])
    
    #model.compile(loss='sparse_categorical_crossentropy', optimizer='nadam', metrics=['mse'])
    #model.fit(allTrainX, allTrainY, epochs=5, batch_size=64, class_weight=class_weights)
    model.fit(allTrainX, allTrainY, epochs=10, batch_size=64, class_weight = 'auto')
    model.summary()
    '''
    
    
    
    #####################test##############################
    
    testPathDir=os.listdir(testFolder)
    allTestX=[]
    allTestY=[]
    allTestZ=[]
    
    allTestsent=[]
    TestBol=True
    
    
    #testPathDir=trainPathDir
    #testFolder=trainFolder
    
    
    
    #allTestX=allTrainX
    #allTestY=allTrainY
    
    
    
    #if target=="oracle":
    #    allTestY=allTestZ
    
    
    #allTestX=np.array(allTestX)
    #allTestY=np.array(allTestY)
    
    #scores = model.evaluate(allTestX, allTestY, verbose=0)
    #print(scores)
    
    
    #############I use svm to try this#####################3
    vector = CountVectorizer(ngram_range=(1,2))
    allTrainX=vector.fit_transform(allTrainsent)
    
    
    clf = svm.SVC(kernel = 'linear')#'linear', 'poly', 'rbf'
    clf.fit(allTrainX,allTrainY)
    #resultList=clf.predict(allTrainX)
    
    ##################################
    
    model = Sequential()
    #model.add(Bidirectional(LSTM(200)))
    model.add(Dense(200,activation='relu'))
    model.add(Dense(200,activation='relu'))
    model.add(Dense(200,activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['mse'])
    
    #model.compile(loss='sparse_categorical_crossentropy', optimizer='nadam', metrics=['mse'])    
    allTrainX=allTrainX
    
    #allTrainX=np.array(allTrainX)
    #allTrainY=np.array(allTrainY)
    
    #model.fit(allTrainX, allTrainY, epochs=10, batch_size=64, class_weight = 'auto')
    #model.summary()
    
    TestBol=True
    allTestX=[]
    allTestY=[]
    allTestZ=[]
    extractAllTrain(allTestX,allTestY,allTestZ,testPathDir,testFolder, allTestsent, TestBol)
    
    allTestX=vector.transform(allTestsent)

    #allTestX=np.array(allTestX)
    
    
    resultList=clf.predict(allTestX)
    #resultList=model.predict(allTestX)
    
    
    if target=="oracle":
        allTestY=allTestZ
    
    TP=0
    TN=0
    FP=0
    FN=0
    
    
    
    
    for i in range(0,len(resultList)-1):
        result=resultList[i]
        if allTestY[i]==1:#real positive
        
            if result>=0.5:#predict positive
                TP=TP+1
            
            if result<0.5:#predict negative
                FN=FN+1
                #show5sents(allTestsent[i], "FN")
                
                
                
        elif allTestY[i]==0:#real negative
            
            if result>=0.5:#predict positive
                FP=FP+1
                #show5sents(allTestsent[i], "FP")
                
            if result<0.5:#predict negative
                TN=TN+1
        
        

    
    print("TP: "+str(TP))
    print("FN: "+str(FN))
    print("FP: "+str(FP))
    print("TN: "+str(TN))
    
    P=float(float(TP)/float(TP+FP))#precision
    R=float(float(TP)/(float(TP + FN)))#recall
    Fscore=float(2*float(P*R)/(float(P+R)));
    Accuracy=float(float(TP + TN)/(float(TP +FP + FN+TN)));
    
    print("Precision: "+str(P))
    print("Recall: "+str(R))
    print("F score: "+str(Fscore))
    print("Accuracy: "+str(Accuracy))            
    print("asd")
            
            


            
            
            
            
            
    

    

