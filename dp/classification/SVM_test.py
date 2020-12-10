import re
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
import os
import random
import shutil
import time
import math
import sys
from gensim.models import Word2Vec
import numpy as np


def show5sents(allTestsent_i, whyText, outputFile):
    
    
    ##########write file##############3
    outputFile.write(os.linesep)
    outputFile.write(os.linesep)
    outputFile.write("---"+whyText+"---"+os.linesep)
    outputFile.write("sent_target:  "+allTestsent_i+os.linesep)
    outputFile.write(os.linesep)
    outputFile.write(os.linesep)
    
    print("")
    print("")
    print("---"+whyText+"---")
    #print("sent1:  "+allTestsent_i[0])
    #print("sent2:  "+allTestsent_i[1])
    #print("sent3:  "+allTestsent_i[2])
    #print("sent4:  "+allTestsent_i[3])
    print("sent_target:  "+allTestsent_i)
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
    #sent5.append(doc2VecTrans(item5))
    sent5.append(word2VecSum(item5))
    
    #if TestBol:
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
    #vector=modelDoc2vec.infer_vector(wordsSentList)
    
    #return vector
    
    
def word2VecSum(oneLine):
    oneLine=oneLine.replace("#step","")
    oneLine=oneLine.replace("#wtep","")
    oneLine=oneLine.replace("#oracle","")
    oneLine=oneLine.strip()
    wordsSentList=re.findall(r"[\w']+|[.,!?;]", oneLine)####this one is without punctuation
    sentVec=np.zeros(500)
    
    
    for word in wordsSentList:
        if word in word_vectors:  
            wordVec=word_vectors[word]
            sentVec=np.sum([sentVec,wordVec],axis=0)
    return sentVec

def extractAllTrain(allTrainX,allTrainY,allTrainZ, pathDir, Folder, allTestsent, TestBol, showtest):#### TestBol and sent5Text are just to use check FP and FN
    
    #testCount=200######3this is just for test
    
    for fileName in pathDir:
        '''
        testCount=testCount-1
        if testCount==0:
            break
        '''
        
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
                #sent5.append(doc2VecTrans(contents[i]))
                sent5.append(word2VecSum(contents[i]))
                fillLabels(contents[i],allTrainY,allTrainZ)#fill labels for title
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
            allTrainX.append(sent5[0])
            
            allTestsent.append(sent5Text[0].replace("#oracle","").replace("#step","").replace("#wtep",""))
            if TestBol:
                showtest.append(sent5Text[0])
                

######main function

if __name__ == '__main__':
    
    fname="./my_word2vec_model_500"
    #modelDoc2vec = Doc2Vec.load(fname)
    modelWord2vec = Word2Vec.load(fname)
    word_vectors = modelWord2vec.wv
    
    
    print(sys.argv[1])
    print(sys.argv[2])
    
    if sys.argv[1]=="-step":
        target="step"
    elif sys.argv[1]=="-oracle":
        target="oracle"
    else:
        print("wrong input")
        exit()
    
    
    if sys.argv[2]=="-google":
        targetFolder="google code"
    elif sys.argv[2]=="-github":
        target="github"
    else:
        print("wrong input")
        exit()
    
    
    '''
    #targetFolder='google code'

    precentAge=40
    
    
    try:
        shutil.rmtree('buildDataSet/'+targetFolder+'/test')
    except:
        print("fail to remove test")
        
        
    try:
        shutil.rmtree('buildDataSet/'+targetFolder+'/train')
    except:
        print("fail to remove train")
        
        
    
    os.makedirs('buildDataSet/'+targetFolder+'/train')
    os.makedirs('buildDataSet/'+targetFolder+'/test')
        
        
    time.sleep(0.5)
    originalFolderName='buildDataSet/'+targetFolder+'/originalFolders'
    pathDir=os.listdir(originalFolderName)
    
    
    countTotal=0
    for subpath in pathDir:
        subpathDir=os.listdir(originalFolderName+'/'+subpath)
        for filename in subpathDir:
            shutil.copyfile(originalFolderName+"/"+subpath+"/"+filename,'buildDataSet/'+targetFolder+"/train/"+filename)
            time.sleep(0.01)
            countTotal=countTotal+1
            #print(countTotal)
            
    
    ##############move to the test#####################
    
    
    countMove=precentAge/100.0*countTotal
    countMove=math.floor(countMove)
    print(countTotal)
    print("countMove")
    print(countMove)
    filesToMove=random.sample(os.listdir('buildDataSet/'+targetFolder+"/train"), int(countMove))
    
    for filename in filesToMove:
        shutil.move('buildDataSet/'+targetFolder+"/train/"+filename, 'buildDataSet/'+targetFolder+"/test/"+filename)
        time.sleep(0.01)
    
    
    
    
    '''
    ##########train############
    #targetFolder='google code'
    
    #target="step"   # you need to configure
    
    trainFolder='buildDataSet/'+targetFolder+'/train'
    testFolder='buildDataSet/'+targetFolder+'/test'
    
    #fname="./my_doc2vec_model_5001"
    
    trainPathDir=os.listdir(trainFolder)
    allTrainX=[]#sentence
    allTrainY=[]#step list
    allTrainZ=[]#crash list
    showtest=[]
    
    allTrainsent=[]
    TestBol=False
    
    extractAllTrain(allTrainX,allTrainY,allTrainZ,trainPathDir,trainFolder, allTrainsent, TestBol, showtest)
    
    if target=="oracle":
        allTrainY=allTrainZ
        
    
    #############I use svm to try this#####################3
    #vector = CountVectorizer(ngram_range=(1,2))
    #allTrainX=vector.fit_transform(allTrainsent)
    print("afterPretrained")
    #####################test##############################
    
    testPathDir=os.listdir(testFolder)
    allTestX=[]
    allTestY=[]
    allTestZ=[]
    
    allTestsent=[]    
    
    
    
    clf = svm.SVC(kernel = 'linear')#'linear', 'poly', 'rbf'
    clf.fit(allTrainX,allTrainY)
    print("aftertrained")
    ##################################
    
    
    TestBol=True
    allTestX=[]
    allTestY=[]
    allTestZ=[]
    showtest=[]
    extractAllTrain(allTestX,allTestY,allTestZ,testPathDir,testFolder, allTestsent, TestBol, showtest)
    
    #allTestX=vector.transform(allTestsent)    
    
    
    if target=="oracle":
        allTestY=allTestZ
    
    resultList=clf.predict(allTestX)    
    
    TP=0
    TN=0
    FP=0
    FN=0
    
    
    outputFile = open("result_"+target+".txt", 'w')
    
    for i in range(0,len(resultList)-1):
        result=resultList[i]
        if allTestY[i]==1:#real positive
        
            if result>=0.5:#predict positive
                TP=TP+1
            
            if result<0.5:#predict negative
                FN=FN+1
                show5sents(showtest[i], "FN", outputFile)
                
                
                
        elif allTestY[i]==0:#real negative
            
            if result>=0.5:#predict positive
                FP=FP+1
                show5sents(showtest[i], "FP", outputFile)
                
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
            
            


            
            
            
            
            
    

    

