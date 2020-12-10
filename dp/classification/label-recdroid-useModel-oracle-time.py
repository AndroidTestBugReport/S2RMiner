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
from gensim.models import Word2Vec
from numpy.random import seed
from tensorflow import set_random_seed
from keras.initializers import Constant
from keras.models import load_model
from keras.models import model_from_json





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
            
    
    sent5.append(word2IDArray(item1))
    sent5.append(word2IDArray(item2))
    sent5.append(word2IDArray(item3))
    sent5.append(word2IDArray(item4))
    sent5.append(word2IDArray(item5))
    
    
    if TestBol:
        sent5Text.append(item1)
        sent5Text.append(item2)
        sent5Text.append(item3)
        sent5Text.append(item4)
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


def word2VecArray(oneLine):
    oneLine=oneLine.replace("#step","")
    oneLine=oneLine.replace("#wtep","")
    oneLine=oneLine.replace("#oracle","")
    oneLine=oneLine.strip()
    wordsSentList=re.findall(r"[\w']+|[.,!?;]", oneLine.lower())####this one is without punctuation
    sentVecArray=np.empty((0,vectorSize))
        
    for word in wordsSentList:
        if word in word_vectors:  
            wordVec=word_vectors[word]
            sentVecArray=np.vstack([sentVecArray,wordVec])
            
        if len(sentVecArray)+1>Maxwordsent:
            print("bingo, exceed the"+str(Maxwordsent)+"words")
            break
            
    for i in range(1,Maxwordsent-len(sentVecArray)+1):
        sentVecArray=np.vstack([sentVecArray,np.zeros(vectorSize)])
    return sentVecArray

def word2IDArray(oneLine):
    oneLine=oneLine.replace("#step","")
    oneLine=oneLine.replace("#wtep","")
    oneLine=oneLine.replace("#oracle","")
    oneLine=oneLine.strip()
    wordsSentList=re.findall(r"[\w']+|[.,!?;]", oneLine.lower())####this one is without punctuation
    #sentVecArray=np.empty((0,vectorSize))
    sentVecArray=np.array([])
    
       
    for word in wordsSentList:
        if word in word2Idx:
            sentVecArray=np.append(sentVecArray,word2Idx[word])
        else:
            sentVecArray=np.append(sentVecArray,word2Idx["UNKNOWN_TOKEN"])
                        
        if len(sentVecArray)+1>Maxwordsent:
            print("bingo, exceed the"+str(Maxwordsent)+"words")
            break
            
    for i in range(1,Maxwordsent-len(sentVecArray)+1):
        sentVecArray=np.append(sentVecArray,0)
        
        
        #sentVecArray=np.vstack([sentVecArray,np.zeros(vectorSize)])
    return sentVecArray
    

    
def word2VecSum(oneLine):
    oneLine=oneLine.replace("#step","")
    oneLine=oneLine.replace("#wtep","")
    oneLine=oneLine.replace("#oracle","")
    oneLine=oneLine.strip()
    wordsSentList=re.findall(r"[\w']+|[.,!?;]", oneLine.lower())####this one is without punctuation
    #sentVec=np.zeros(300)
    sentVec=np.array([])
    
    for word in wordsSentList:
        if word in word_vectors:  
            wordVec=word_vectors[word]
            sentVec=np.hstack([sentVec,wordVec])
            
        if len(sentVec)+50>vectorSize:
            print("bingo, exceed the 20 words")
            break
            
    sentVec=np.pad(sentVec, (0, vectorSize-len(sentVec)), 'constant')
    return sentVec
    
def generateLabel(model, testPathDir, testFolder, targetFolder):
    for fileName in testPathDir:
        print(fileName)
        f = open(testFolder+"/"+fileName)
        
        outputFile = open(targetFolder+"/"+fileName, 'w')
        
        
        
        contents = f.readlines()
        
        titleTag=False
        commentTag=0
        
        for i in range(0,len(contents)-1):
            line=contents[i]
            
            sent5=[]
            sent5Text=[]
            if "#####title#12345#" in line:
                
                outputFile.write("#####title#12345#"+os.linesep)
                
                
                if i==0:
                    titleTag=True#this is the title
                    continue
                else:
                    titleTag=False
                    commentTag=1#it means the first sentence of a somment
                    continue
            
            if titleTag==True:#this is for the title
                sent5.append(word2IDArray("#####title#12345#"))
                sent5.append(word2IDArray("#####title#12345#"))
                sent5.append(word2IDArray("#####title#12345#"))
                sent5.append(word2IDArray("#####title#12345#"))
                sent5.append(word2IDArray(contents[i]))
                
                
                if TestBol:
                    sent5Text.append("#####title#12345#")
                    sent5Text.append("#####title#12345#")
                    sent5Text.append("#####title#12345#")
                    sent5Text.append("#####title#12345#")
                    sent5Text.append(contents[i])
                
                
                
                
            elif commentTag>0:#this is for the comments
                if "####comment#12345#" in line:
                    outputFile.write("####comment#12345#"+os.linesep)
                    
                    
                    commentTag=1
                    continue
                else:
                    sent5=fillComments(contents,i,TestBol,sent5Text)#### TestBol and sent5Text are just to use check FP and FN
                    commentTag=commentTag+1
                    
                
                
                
                #fillLabels(contents[i],allTrainY,allTrainZ)#fill labels for comments 
                
            predictResult=model.predict(np.array([sent5]))
            
            oneLine=contents[i]
            oneLine=oneLine.replace("#step","")
            oneLine=oneLine.replace("#wtep","")
            oneLine=oneLine.replace("#oracle","")
            oneLine=oneLine.strip()
            
            if predictResult>0.5: 
                oneLine=oneLine+" #oracle"
                
            outputFile.write(oneLine+os.linesep)
            
        outputFile.write("####comment#12345#"+os.linesep)
        outputFile.close()
                
    

def extractAllTrain(allTrainX,allTrainY,allTrainZ, pathDir, Folder, allTestsent, TestBol):#### TestBol and sent5Text are just to use check FP and FN
    
    #testCount=50######3this is just for test
    
    for fileName in pathDir:
        
        
        print(fileName)
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
                sent5.append(word2IDArray("#####title#12345#"))
                sent5.append(word2IDArray("#####title#12345#"))
                sent5.append(word2IDArray("#####title#12345#"))
                sent5.append(word2IDArray("#####title#12345#"))
                sent5.append(word2IDArray(contents[i]))
                
                fillLabels(contents[i],allTrainY,allTrainZ)#fill labels for title
                
                if TestBol:
                    sent5Text.append("#####title#12345#")
                    sent5Text.append("#####title#12345#")
                    sent5Text.append("#####title#12345#")
                    sent5Text.append("#####title#12345#")
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
                allTestsent.append(sent5Text)




######main function

if __name__ == '__main__':
    seed(1)
    set_random_seed(2)
    
    
    fname="./my_word2vec_model_50"
    modelWord2vec = Word2Vec.load(fname)
    word_vectors = modelWord2vec.wv
    
    
    Maxwordsent=200
    vectorSize=50
    ##############
    
    
    word2Idx = {}
    wordEmbeddings = []#22949
    
    word2Idx=np.load('word2Idx_oracle.npy', allow_pickle=True)
    word2Idx=word2Idx.item()
    
    
    ##########train############
    targetFolder='google code'
    
    target="oracle"
    
    trainFolder='buildDataSet/'+'recdroid-train'
    testFolder='buildDataSet/'+'recdroid-test-time'
    
    
    model=load_model("keras_model_oracle.h5")
    model.summary()
    #####################test##############################
    
    testPathDir=os.listdir(testFolder)
    allTestX=[]
    allTestY=[]
    allTestZ=[]
    
    allTestsent=[]
    TestBol=True
    
    
    extractAllTrain(allTestX,allTestY,allTestZ,testPathDir,testFolder, allTestsent, TestBol)
    
    if target=="oracle":
        allTestY=allTestZ
    
    
    allTestX=np.array(allTestX)
    allTestY=np.array(allTestY)
    
    allTestX=allTestX.astype(int)
    allTestY=allTestY.astype(int)
    
    #scores = model.evaluate(allTestX, allTestY, verbose=0)
    #print(scores)
    
    
    
    resultList=model.predict(allTestX)
    
    targetFolder="recdroid-labeled-by-model/oracle"
    generateLabel(model, testPathDir, testFolder, targetFolder)
    
    
    
    
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
                show5sents(allTestsent[i], "FN")
                
                
                
        elif allTestY[i]==0:#real negative
            
            if result>=0.5:#predict positive
                FP=FP+1
                show5sents(allTestsent[i], "FP")
                
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
            
        
    
    
    #model.predict(X_train[0]) # this predict needs the first dimension of the X_train
            
    print("asd")
            
    
    outputFile = open("result_test_muli.txt", 'a')
    outputFile.write("\n")
    outputFile.write("original embeding_cnn_lstm\n")
    outputFile.write("Precision:  "+str(P) +"\n")
    outputFile.write("Precision:  "+str(R) +"\n")
    outputFile.write("Precision:  "+str(Fscore) +"\n")
    outputFile.write("Precision:  "+str(Accuracy) +"\n")
    outputFile.close()