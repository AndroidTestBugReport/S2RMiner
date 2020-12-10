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
from keras.preprocessing.sequence import pad_sequences




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



def fillComments(sentences, i, TestBol,sent5Text, sent5char, sent5Command):
    sent5=[]#this is for return
    
    
    item1=""#2 before sentence
    item2=""#2 behind sentence
    item3=""#1 before sentence
    item4=""#1 behind sentence
    item5=""#current sentence
    
    
    text1=""
    text2=""
    text3=""
    text4=""
    text5=""
    
    #########current sentences
    item5=sentences[i][0]
    
    
    #########fill before sentences
    lineBefore=3
    
    
    for kk in range(1,3):#[1,2] check where is the comment break, record as lineBefore
        if "####comment#12345#" in sentences[i-kk][0] or "#####title#12345#" in sentences[i-kk][0]:
           lineBefore=kk
           break
    
    if lineBefore==3:
        item1=sentences[i-2][1]
        item3=sentences[i-1][1]
        
        text1

    if lineBefore==2:
        item1=sent5Command
        item3=sentences[i-1][1]
        
        
    if lineBefore==1:
        item1=sent5Command
        item3=sent5Command
        
           
    ##########fill behind sentences
    lineBehind=3
    
    for kk in range(1,3):#[1,2] check where is the comment break, record as lineBefore
        if "####comment#12345#" in sentences[i+kk][0] or "#####title#12345#" in sentences[i+kk][0]:
           lineBehind=kk
           break
    
    
    if lineBehind==3:
        item2=sentences[i+2][1]
        item4=sentences[i+1][1]

    
    if lineBehind==2:
        item2=sent5Command
        item4=sentences[i+1][1]
    
    if lineBehind==1:
        item2=sent5Command
        item4=sent5Command
            
    
    sent5.append(item1)
    sent5.append(item2)
    sent5.append(item3)
    sent5.append(item4)
    sent5.append(item5)
    
    
    
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
    wordsSentList=re.findall(r"[\w']+|[.,!?;]", oneLine.lower())####this one is without punctuation
    vector=modelDoc2vec.infer_vector(wordsSentList)
    
    return vector

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

def word2IDArray(oneLine, sent5char):
    oneLine=oneLine.replace("#step","")
    oneLine=oneLine.replace("#wtep","")
    oneLine=oneLine.replace("#oracle","")
    oneLine=oneLine.strip()
    wordsSentList=re.findall(r"[\w']+|[.,!?;]", oneLine)####this one is without punctuation
    #sentVecArray=np.empty((0,vectorSize))
    
    sentVecArray=np.array([])
    
    maxlen = 52
    charDoubleArray=np.empty((0,maxlen))
    
    
    for word in wordsSentList:
        if word in word2Idx:
            sentVecArray=np.append(sentVecArray,word2Idx[word.lower()])
        else:
            sentVecArray=np.append(sentVecArray,word2Idx["UNKNOWN_TOKEN"])
                        
        if len(sentVecArray)+1>Maxwordsent:
            print("bingo, exceed the"+str(Maxwordsent)+"words")
            break
            
        ###########word2char part###########
        charVecArray=np.array([])
        for c in word:
            charVecArray=np.append(charVecArray,char2Idx[c])
            
            if len(charVecArray)+1>maxlen:
                print("bingo, exceed the"+str(maxlen)+"chars")
                break
            
            
        charVecArray=np.pad(charVecArray,(0, maxlen-len(charVecArray)),'constant')
        charDoubleArray=np.vstack([charDoubleArray,charVecArray])
        
        
        
    for i in range(1,Maxwordsent-len(sentVecArray)+1):
        charDoubleArray=np.vstack([charDoubleArray,np.zeros(maxlen)])
            
            
            
            
    for i in range(1,Maxwordsent-len(sentVecArray)+1):
        sentVecArray=np.append(sentVecArray,0)
        
    
    sent5char.append(charDoubleArray)
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
    
    

def extractAllTrainnew(allTrainX,allTrainY,allTrainZ, trainSent, allTestsent, TestBol, allCharTrain):#### TestBol and sent5Text are just to use check FP and FN
    
    
    sent5charTitile=[]
    sent5Title=word2IDArray("#####title#12345#", sent5charTitile)
    
    sent5charCommand=[]
    sent5Command=word2IDArray("####comment#12345#", sent5charCommand)
    
    
    
    for sentences in files:
        titleTag=False
        commentTag=0
        
        for i, sentence in enumerate(sentences):
            line=sentence[0]
            
            sent5=[]
            sent5Text=[]
            sent5char=[]
            if "#####title#12345#" in line:
                if i==0:
                    titleTag=True#this is the title
                    continue
                else:
                    titleTag=False
                    commentTag=1#it means the first sentence of a somment
                    continue
            
            if titleTag==True:#this is for the title
                sent5.append(sent5Title)
                sent5.append(sent5Title)
                sent5.append(sent5Title)
                sent5.append(sent5Title)
                sent5.append(sentence[1])
                
                allTrainY.append(sentence[3][0])
                allTrainZ.append(sentence[3][1])
                
                if TestBol:
                    sent5Text.append("#####title#12345#")
                    sent5Text.append("#####title#12345#")
                    sent5Text.append("#####title#12345#")
                    sent5Text.append("#####title#12345#")
                    sent5Text.append(line)
                
                
            elif commentTag>0:#this is for the comments
                if "####comment#12345#" in line:
                    commentTag=1
                    continue
                else:
                    sent5=fillComments(sentences,i,TestBol,sent5Text, sent5char, sent5Command,)#### TestBol and sent5Text are just to use check FP and FN
                    commentTag=commentTag+1
                
                allTrainY.append(sentence[3][0])
                allTrainZ.append(sentence[3][1])
                
                #sent5=fillComments(contents,i,TestBol,sent5Text)  7.18 I do not know why do I add this before, so I remove it
            allTrainX.append(sent5)
            allCharTrain.append(sent5char)
            if TestBol:
                allTestsent.append(sent5Text)

    
    
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
            sent5char=[]
            if "#####title#12345#" in line:
                if i==0:
                    titleTag=True#this is the title
                    continue
                else:
                    titleTag=False
                    commentTag=1#it means the first sentence of a somment
                    continue
            
            if titleTag==True:#this is for the title
                sent5.append(word2IDArray("#####title#12345#", sent5char))
                sent5.append(word2IDArray("#####title#12345#", sent5char))
                sent5.append(word2IDArray("#####title#12345#", sent5char))
                sent5.append(word2IDArray("#####title#12345#", sent5char))
                sent5.append(word2IDArray(contents[i], sent5char))
                
                
                
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
                    sent5=fillComments(contents,i,TestBol,sent5Text, sent5char)#### TestBol and sent5Text are just to use check FP and FN
                    commentTag=commentTag+1
                    
                fillLabels(contents[i],allTrainY,allTrainZ)#fill labels for comments 
                #sent5=fillComments(contents,i,TestBol,sent5Text)  7.18 I do not know why do I add this before, so I remove it
            allTrainX.append(sent5)
            allCharTrain.append(sent5char)
            if TestBol:
                allTestsent.append(sent5Text)


#################################################new method##########################




def word2Array(oneLine):
    oneLine=oneLine.replace("#step","")
    oneLine=oneLine.replace("#wtep","")
    oneLine=oneLine.replace("#oracle","")
    oneLine=oneLine.strip()
    wordsSentList=re.findall(r"[\w']+|[.,!?;]", oneLine)####this one is without punctuation    
    
    wordIdArray=np.array([])
    
    for word in wordsSentList:
        if word in word2Idx:
            wordIdArray=np.append(wordIdArray,word2Idx[word.lower()])
        else:
            wordIdArray=np.append(wordIdArray,word2Idx["UNKNOWN_TOKEN"])
                        
        if len(wordIdArray)+1>Maxwordsent:
            print("bingo, exceed the"+str(Maxwordsent)+"words")
            break
            
            
    for i in range(1,Maxwordsent-len(wordIdArray)+1):
        wordIdArray=np.append(wordIdArray,0)
        
    return wordIdArray
        
        
def char2Array(oneLine):    
    oneLine=oneLine.replace("#step","")
    oneLine=oneLine.replace("#wtep","")
    oneLine=oneLine.replace("#oracle","")
    oneLine=oneLine.strip()
    wordsSentList=re.findall(r"[\w']+|[.,!?;]", oneLine)####this one is without punctuation    
    
    maxlen = 52
    charIdDoubleArray=np.empty((0,maxlen))
    
    
    for word in wordsSentList:            
        ###########word2char part###########
        charVecArray=np.array([])
        for c in word:
            charVecArray=np.append(charVecArray,char2Idx[c])
            
            if len(charVecArray)+1>maxlen:
                print("bingo, exceed the"+str(maxlen)+"chars")
                break
            
            
        charVecArray=np.pad(charVecArray,(0, maxlen-len(charVecArray)),'constant')
        charIdDoubleArray=np.vstack([charIdDoubleArray,charVecArray])
        
        
        
    for i in range(1,Maxwordsent-len(charIdDoubleArray)+1):
        charIdDoubleArray=np.vstack([charIdDoubleArray,np.zeros(maxlen)])        
    
    return charIdDoubleArray

def takeTag(line):
    #[step,oracle]
    tag=[0,0]
    
    if "#step" in line or "#wtep" in line:
        tag[0]=1
    else:
        tag[0]=0
        
    #########this is for #oracle label############
    if "#oracle" in line:
        tag[1]=1
    else:
        tag[1]=0
    return tag

def readFile(pathDir, Folder):
    
    files=[]
    
    for fileName in pathDir:
        f = open(Folder+"/"+fileName)
        contents = f.readlines()
        
        sentences = []
        sentence = []
        
        for i,line in enumerate(contents):
            
            
            wordIdArray=word2Array(line)
            charIdDoubleArray=char2Array(line)
            tag=takeTag(line)
            sentences.append([line, wordIdArray, charIdDoubleArray, tag])
        files.append(sentences)
    return files
        
    


######main function

if __name__ == '__main__':
    seed(1)
    set_random_seed(2)
    
    
    fname="./my_word2vec_model_50"
    #modelDoc2vec = Doc2Vec.load(fname)
    modelWord2vec = Word2Vec.load(fname)
    word_vectors = modelWord2vec.wv
    
    
    Maxwordsent=200
    vectorSize=50
    ##############
    
    
    word2Idx = {}
    char2Idx = {}
    wordEmbeddings = []#22949
    
    for word in word_vectors.vocab:
        
        if len(word2Idx) == 0: #Add padding+unknown
            word2Idx["PADDING_TOKEN"] = len(word2Idx)
            vector = np.zeros(vectorSize) #Zero vector vor 'PADDING' word
            wordEmbeddings.append(vector)
            
            word2Idx["UNKNOWN_TOKEN"] = len(word2Idx)
            vector = np.random.uniform(-0.25, 0.25, vectorSize)
            wordEmbeddings.append(vector)
        word2Idx[word]=len(word2Idx)
        wordEmbeddings.append(word_vectors[word])
    
    
    wordEmbeddings = np.array(wordEmbeddings)
    ##############
    for c in " 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,-_()[]{}!?:;#'\"/\\%$`&=*+@^~|":
        char2Idx[c] = len(char2Idx)#dict: {'1': 4, ' ': 2, 'UNKNOWN': 1, 'PADDING': 0, '0': 3, '3': 6, '2': 5} len(char2Idx)=95
    
    
    ##########train############
    targetFolder='google code'
    
    target="step"
    
    trainFolder='buildDataSet/'+targetFolder+'/train'
    testFolder='buildDataSet/'+targetFolder+'/test'
    trainPathDir=os.listdir(trainFolder)
    files=readFile(trainPathDir, trainFolder)
    
    
    
    
    
    
    
    
    
    allTrainX=[]#sentence
    allTrainY=[]#step list
    allTrainZ=[]#crash list
    
    allCharTrain=[]
    allTrainsent=[]
    
    TestBol=False
    extractAllTrainnew(allTrainX, allTrainY,allTrainZ,trainSent, allTrainsent, TestBol, allCharTrain)
    
    
    
    
    
    
    
    
    
    TestBol=False
    
    extractAllTrain(allTrainX,allTrainY,allTrainZ,trainPathDir,trainFolder, allTrainsent, TestBol, allCharTrain)
    
    if target=="oracle":
        allTrainY=allTrainZ
    
    
    allTrainX=np.array(allTrainX)
    allTrainY=np.array(allTrainY)
    allCharTrain=np.array(allCharTrain)
    
    allTrainX=allTrainX.astype(int)
    allTrainY=allTrainY.astype(int)
    allCharTrain=allCharTrain.astype(int)
    
    print("afterPretrained")
    
    #class_weights = class_weight.compute_class_weight('balanced', np.unique(allTrainY), allTrainY)
    
    
    #class_weights = {0: 1., 1: 5.}
    
    
    '''
    ##########here is for smote#######################
    
    
    #allTrainX=allTrainX.reshape
    
    allTrainX=allTrainX.reshape(-1,5*vectorSize)
    
    sm = SMOTE()
    allTrainX, allTrainY = sm.fit_resample(allTrainX, allTrainY)
    
    allTrainX=allTrainX.reshape(-1,5,vectorSize)
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
    
    sents_input = Input(shape=(5,Maxwordsent), name='sents_input')
    #words = TimeDistributed(Embedding(input_dim=wordEmbeddings.shape[0], output_dim=wordEmbeddings.shape[1],  weights=[wordEmbeddings], trainable=False))(sents_input)#shape[0]is the vacoabilry size, shape[1] is output size #embedding_1 (Embedding)         (None, None, 100)  the first None is the batch, the second None is the number of word in a sentence
    words = TimeDistributed(Embedding(input_dim=wordEmbeddings.shape[0], output_dim=wordEmbeddings.shape[1], embeddings_initializer=Constant(wordEmbeddings), trainable=False))(sents_input)#shape[0]is the vacoabilry size, shape[1] is output size #embedding_1 (Embedding)         (None, None, 100)  the first None is the batch, the second None is the number of word in a sentence
        
    conv1d_out= TimeDistributed(Conv1D(kernel_size=3, filters=Maxwordsent, padding='same',activation='tanh', strides=1))(words) #(None, None, 52, 30) filters is the kernel number, it decide output how many vecters
    maxpool_out=TimeDistributed(MaxPooling1D(Maxwordsent))(conv1d_out)#in every items in 30, pick the max 52 the output is (None, None, 1, 30)
    #maxpool_out=MaxPooling1D(52, name='aaa')(conv1d_out)#in every items in 30, pick the max 52 the output is (None, None, 1, 30)
    char = TimeDistributed(Flatten())(maxpool_out)#(TimeDistrib (None, None, 500)   #TimeDistributed targets the time, it can accept the None.
    output = Dropout(0.5)(char)#(None, None, 500)
    
    LSTM_output=Bidirectional(LSTM(vectorSize, dropout=0.50, recurrent_dropout=0.25))(output)
    Dense_output=Dense(vectorSize,activation='relu')(LSTM_output)
    Dropout_output=Dropout(0.5)(Dense_output)
    output=Dense(1, activation='sigmoid')(Dropout_output)
    model=Model(inputs=[sents_input],outputs=[output])
    model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['mse'])
    
    model.summary()
    
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
    model.fit(allTrainX, allTrainY, epochs=30, batch_size=64, class_weight = 'auto')
    model.summary()
    
    
    
    
    #####################test##############################
    
    testPathDir=os.listdir(testFolder)
    allTestX=[]
    allTestY=[]
    allTestZ=[]
    allCharTest=[]
    
    allTestsent=[]
    TestBol=True
    
    
    #testPathDir=trainPathDir
    #testFolder=trainFolder
    
    extractAllTrain(allTestX,allTestY,allTestZ,testPathDir,testFolder, allTestsent, TestBol, allCharTest)
    
    
    
    if target=="oracle":
        allTestY=allTestZ
    
    
    allTestX=np.array(allTestX)
    allTestY=np.array(allTestY)
    
    allTestX=allTestX.astype(int)
    allTestY=allTestY.astype(int)
    
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
            
    
    outputFile = open("result_test_muli.txt", 'a')
    outputFile.write("\n")
    outputFile.write("original embeding_cnn_lstm\n")
    outputFile.write("Precision:  "+str(P) +"\n")
    outputFile.write("Precision:  "+str(R) +"\n")
    outputFile.write("Precision:  "+str(Fscore) +"\n")
    outputFile.write("Precision:  "+str(Accuracy) +"\n")
    outputFile.close()


            
            
            
            
            
    

    

