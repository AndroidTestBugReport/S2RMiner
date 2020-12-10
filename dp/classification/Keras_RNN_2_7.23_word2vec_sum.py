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
            
    
    sent5.append(word2VecSum(item1))
    sent5.append(word2VecSum(item2))
    sent5.append(word2VecSum(item3))
    sent5.append(word2VecSum(item4))
    sent5.append(word2VecSum(item5))
    
    
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

def doc2VecTrans(oneLine):
    
    oneLine=oneLine.replace("#step","")
    oneLine=oneLine.replace("#wtep","")
    oneLine=oneLine.replace("#oracle","")
    oneLine=oneLine.strip()
            #wordList=oneLine.split(" ")###this one is with punctuation
    wordsSentList=re.findall(r"[\w']+|[.,!?;]", oneLine)####this one is without punctuation
    vector=modelDoc2vec.infer_vector(wordsSentList)
    
    return vector
    
    
def word2VecSum(oneLine):
    oneLine=oneLine.replace("#step","")
    oneLine=oneLine.replace("#wtep","")
    oneLine=oneLine.replace("#oracle","")
    oneLine=oneLine.strip()
    wordsSentList=re.findall(r"[\w']+|[.,!?;]", oneLine)####this one is without punctuation
    sentVec=np.zeros(vectorsize)
    
    
    for word in wordsSentList:
        if word in word_vectors:  
            wordVec=word_vectors[word]
            sentVec=np.sum([sentVec,wordVec],axis=0)
    if not len(wordsSentList)==0:
        sentVec=np.divide(sentVec, len(wordsSentList))
    return sentVec
    
    

def extractAllTrain(allTrainX,allTrainY,allTrainZ, pathDir, Folder, allTestsent, TestBol):#### TestBol and sent5Text are just to use check FP and FN
    
    #testCount=100######3this is just for test
    
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
                sent5.append(word2VecSum("#####title#12345#"))
                sent5.append(word2VecSum("#####title#12345#"))
                sent5.append(word2VecSum("#####title#12345#"))
                sent5.append(word2VecSum("#####title#12345#"))
                sent5.append(word2VecSum(contents[i]))
                
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
    
    vectorsize=500
    ##########train############
    targetFolder='google code'
    
    target="step"
    
    trainFolder='buildDataSet/'+targetFolder+'/train'
    testFolder='buildDataSet/'+targetFolder+'/test'
    
    fname="./my_word2vec_model"
    #modelDoc2vec = Doc2Vec.load(fname)
    modelWord2vec = Word2Vec.load(fname)
    word_vectors = modelWord2vec.wv
    '''
    if 'word' in word_vectors.vocab:
        print("bingo")
    
    print("not")
    '''
    
    trainPathDir=os.listdir(trainFolder)
    allTrainX=[]#sentence
    allTrainY=[]#step list
    allTrainZ=[]#crash list
    
    allTrainsent=[]
    TestBol=False
    
    extractAllTrain(allTrainX,allTrainY,allTrainZ,trainPathDir,trainFolder, allTrainsent, TestBol)
    
    if target=="oracle":
        allTrainY=allTrainZ
    
    
    allTrainX=np.array(allTrainX)
    allTrainY=np.array(allTrainY)
    
    print("afterPretrained")
    
    #class_weights = class_weight.compute_class_weight('balanced', np.unique(allTrainY), allTrainY)
    
    
    #class_weights = {0: 1., 1: 5.}
    
    
    
    ##########here is for smote#######################
    
    
    #allTrainX=allTrainX.reshape
    '''
    allTrainX=allTrainX.reshape(-1,5*vectorsize)
    
    sm = SMOTE()
    allTrainX, allTrainY = sm.fit_resample(allTrainX, allTrainY)
    
    allTrainX=allTrainX.reshape(-1,5,vectorsize)
    '''
    
    ####I use a test sequence as 
    ####a=np.array([1,2,3,4,5,6,7,8])
    ###b=a.reshape(-1,2,2)
    ###b.reshape(-1,2)
    ###b.reshape(-1,4)
    
    ##################################################
    
    
    
    '''
    words_input = Input(shape=(None,),dtype='int32',name='words_input')
    words = Embedding(input_dim=wordEmbeddings.shape[0], output_dim=wordEmbeddings.shape[1],  weights=[wordEmbeddings], trainable=False)(words_input)
    casing_input = Input(shape=(None,), dtype='int32', name='casing_input')
    casing = Embedding(output_dim=caseEmbeddings.shape[1], input_dim=caseEmbeddings.shape[0], weights=[caseEmbeddings], trainable=False)(casing_input)
    character_input=Input(shape=(None,52,),name='char_input')
    embed_char_out=TimeDistributed(Embedding(len(char2Idx),30,embeddings_initializer=RandomUniform(minval=-0.5, maxval=0.5)), name='char_embedding')(character_input)
    dropout= Dropout(0.5)(embed_char_out)
    conv1d_out= TimeDistributed(Conv1D(kernel_size=3, filters=30, padding='same',activation='tanh', strides=1))(dropout)
    maxpool_out=TimeDistributed(MaxPooling1D(52))(conv1d_out)
    char = TimeDistributed(Flatten())(maxpool_out)
    char = Dropout(0.5)(char)
    output = concatenate([words, casing,char])
    output = Bidirectional(LSTM(200, return_sequences=True, dropout=0.50, recurrent_dropout=0.25))(output)
    output = TimeDistributed(Dense(len(label2Idx), activation='softmax'))(output)
    model = Model(inputs=[words_input, casing_input,character_input], outputs=[output])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='nadam')
    model.summary()
    '''
    
    sents_input = Input(shape=(5,vectorsize), name='sents_input')
    LSTM_output=Bidirectional(LSTM(vectorsize, dropout=0.50, recurrent_dropout=0.25))(sents_input)
    Dense_output=Dense(vectorsize,activation='relu')(LSTM_output)
    Dropout_output=Dropout(0.5)(Dense_output)
    output=Dense(1, activation='sigmoid')(Dropout_output)
    model=Model(inputs=[sents_input],outputs=[output])
    model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['mse'])
    
    
    '''
    model = Sequential()
    #model.add(LSTM(100, batch_input_shape=(5, 500))) #(None, 100)
    #model.add(LSTM(200, input_shape=(5,200))) #(None, 100)    
    
    #model.add(LSTM(200, input_shape=(5,200), return_sequences=True, dropout=0.50, recurrent_dropout=0.25))
    #model.add(LSTM(200, go_backwards=True, dropout=0.50, recurrent_dropout=0.25))
    model.add(Bidirectional(LSTM(200, dropout=0.50, recurrent_dropout=0.25)))
    
    model.add(Dense(200,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['mse'])
    '''
    
    #model.compile(loss='sparse_categorical_crossentropy', optimizer='nadam', metrics=['mse'])
    #model.fit(allTrainX, allTrainY, epochs=5, batch_size=64, class_weight=class_weights)
    model.fit(allTrainX, allTrainY, epochs=5, batch_size=64, class_weight = 'auto')
    model.summary()
    
    
    
    
    #####################test##############################
    
    testPathDir=os.listdir(testFolder)
    allTestX=[]
    allTestY=[]
    allTestZ=[]
    
    allTestsent=[]
    TestBol=True
    
    
    #testPathDir=trainPathDir
    #testFolder=trainFolder
    
    extractAllTrain(allTestX,allTestY,allTestZ,testPathDir,testFolder, allTestsent, TestBol)
    
    
    
    if target=="oracle":
        allTestY=allTestZ
    
    
    allTestX=np.array(allTestX)
    allTestY=np.array(allTestY)
    
    #scores = model.evaluate(allTestX, allTestY, verbose=0)
    #print(scores)
    
    
    
    resultList=model.predict(allTestX)
    
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
            
            


            
            
            
            
            
    

    

