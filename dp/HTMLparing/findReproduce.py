'''
Created on Oct 19, 2017

@author: yu
'''
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer  
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfTransformer  
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import os
import sys
from xml.dom.minidom import Document
from xml.dom import minidom
from spacy.en import English
from sklearn.externals import joblib
import scipy.sparse as sp
import copy
from lxml import html
from lxml import etree
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

def replaceNumDot(x):#change 1. to numbDot
    index=x.find(". ")
    if x[index-1].isdigit():
        x=x.replace(x[index-1]+"." , "numDot")

    return x


def processLine(x,senList, stepList, oracleList):#process one line from html, one line may have more than one sentence
    index=x.find(". ")
    if index==-1: #there is no any ". "left in the sentence
        if x!="":
            
            if x.find("#step")!=-1:# x.contains #step# such as a b c d e #step#
                stepList.append(1)
                x=x.replace("#step","")
            else:
                stepList.append(0)

        
        
            
            if x.find("#oracle")!=-1:# x.contains #oracle#
                oracleList.append(1)
                x=x.replace("#oracle","")
            else:
                oracleList.append(0)
            senList.append(x.strip())
        return
    
    
    
    if x[0: index+1]!="" and not x[index-1].isdigit() and not x[index-1]==".":
        add=x[0: index+1];
        add.replace("#step","")
        add.replace("#oracle","")
        senList.append(x[0: index+1].strip())
        
        x=x[x.find(". ")+1:]#cut the x
        
        
        stepIndex=x.find("#step")
        if stepIndex>=0 and stepIndex < 3:#check step
            x=x[6:]
            stepList.append(1)
        else:
            stepList.append(0)
            
        oracleIndex=x.find("#oracle")
        if oracleIndex>=0 and oracleIndex < 3:#check oracle
            x=x[8:]
            oracleList.append(1)
        else:
            oracleList.append(0)
            
    elif x[index-1].isdigit():
        x=x.replace(x[index-1]+"." , "numDot")
    elif x[index-1]==".":
        x=x.replace(x[index-1]+". " , "soOnDot ")
    processLine(x,senList, stepList, oracleList)

def fillmatrix(subpathDir,x,y,z,allDir):
    #index=0
    #if allDir=="firefox":
    
    
    index=0
    for suballDir in subpathDir:
            
            sentenceList=[]
            if allDir=="google code":          
                tree = html.parse("labeled/"+allDir+"/"+suballDir)
                
                buyers = tree.xpath('//markdown-widget[@text="comment.content"]')


                sentenceList=buyers[0].text_content().splitlines()
                   
            else:
                tree = html.parse("labeled/"+allDir+"/"+suballDir)
                buyers = tree.xpath('//td')
                sentenceList=buyers[0].text_content().splitlines()
            
            for sentence in sentenceList:
                if not len(sentence.split())==0:
                    processLine(sentence,x, y, z)
                    
def testfillmatrix(subpathDir,x,y,z,allDir):
    #index=0
    #if allDir=="firefox":
    
    
    index=0
    for suballDir in subpathDir:
            
            sentenceList=[]
            if allDir=="google code":          
                tree = html.parse("test/"+allDir+"/"+suballDir)
                buyers = tree.xpath('//markdown-widget[@text="comment.content"]')
                sentenceList=buyers[0].text_content().splitlines()
                    
            else:
                tree = html.parse("test/"+allDir+"/"+suballDir)
                buyers = tree.xpath('//td')
                sentenceList=buyers[0].text_content().splitlines()
            
            for sentence in sentenceList:
                if not len(sentence.split())==0:
                    if "I rotate my phone." in sentence:
                        print("bingo")
                    processLine(sentence,x, y, z)

            
def fillPosDep(x,pos,dep,stem):
                        
            for item in x:
                posStr=""
                depStr=""
                stemStr=""

                for sent in nlp(unicode(item)).sents:
                    sentnlp=nlp(unicode(str(sent)))
                    for word in sentnlp:
                        stemStr=stemStr+" "+word.lemma_
                        posStr=posStr+" "+word.pos_
                        depStr=depStr+" "+word.dep_
                stem.append(stemStr)
                pos.append(posStr)
                dep.append(depStr)

                


if __name__ == '__main__':
    
    #-k "ob" -t "t"
    
    keyword=sys.argv[2]#train which part of the bug report,# "eb": expected behavior,"ob":observe behavior, "sr":steps to recovery 
    trainorload=sys.argv[4]#if train "t", if load "l" 
    
    
    #keyword="ob"#train which part of the bug report,# "eb": expected behavior,"ob":observe behavior, "sr":steps to recovery 
    
    reload(sys)
    sys.setdefaultencoding('utf-8')
    '''
    nlp=English();
    sentence=u"During multiple input of identical values e.g. 10 km (by day) and 10 litres some calculations fail. E.g. first calculates correcty 100 l/km but then next two results are \"infinity l / 100 km. Plotting will death lock program then and you have to restart device.";
    
    for sent in nlp(sentence).sents:
        doc=nlp(unicode(sent))
        for word in doc:
            print(word.text, word.pos_, word.dep_)
    '''
    global nlp
    nlp=English();
    if(trainorload=="t"):
        #nlp=English();
        x=[]#x is sentence
        y=[]#y is step
        z=[]#z is oracle
        pos=[]
        dep=[]
        stem=[]
        pathDir=os.listdir("labeled")
        for allDir in pathDir:
            #if(allDir=="github"):
            #    continue#the firefox is the test set
            subpathDir=os.listdir("labeled/"+allDir)
            fillmatrix(subpathDir,x,y,z,allDir)
        fillPosDep(x,pos,dep,stem)
        #print("x length: "+str(len(x)))      
        #print("y length: "+str(len(y))) 
        vector = CountVectorizer(ngram_range=(1,3))#ngram_range=(1,2)#gram 1 and gram 2# the gram 3 will be very costly
        vectorpos=CountVectorizer(ngram_range=(1,1))
        vectordep=CountVectorizer(ngram_range=(1,1))

        
        x=vector.fit_transform(stem)#.toarray()
        pos=vectorpos.fit_transform(pos)#.toarray()
        dep=vectordep.fit_transform(dep)#.toarray()
        
        #x=sp.hstack((x, pos), format='csr')
        x=sp.hstack((x, dep), format='csr')#test to remove
        
        #print(pos)
        #print(len(x.toarray()))
        #print(len(pos.toarray()))
        #print(len(dep.toarray()))
        
        #x=np.concatenate((x,pos),axis=1)#merge these three array
        #x=np.concatenate((x,dep),axis=1)#merge these three array    
        
        
        clf = svm.SVC(kernel = 'linear')#'linear', 'poly', 'rbf'
        #clf = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=2)
        
        #clf = GaussianNB()
        clf.fit(x,y)        
        
        
        print("trainfinish")
   
        
        #######test  ########################################
        ##############################################
        ######################################3
        #####################3
        ##############33
        testx=[]
        testy=[]
        testz=[]
        testpos=[]
        testdep=[]
        teststem=[]
        
        x=[]#x is sentence
        y=[]#y is step
        z=[]#z is oracle
        pos=[]
        dep=[]
        pathDir=os.listdir("test")
        for allDir in pathDir:
            #if(allDir=="github"):
            #    continue#the firefox is the test set
            subpathDir=os.listdir("test/"+allDir)
            testfillmatrix(subpathDir,testx,testy,testz,allDir)
        fillPosDep(testx,testpos,testdep,teststem)
        
        
        
        global indexTable
        indexTable={}#key is the index of test sentence, value is the filename
        global realTable
        realTable={}#key is the file name, value is the label
        global predictTable
        predictTable={}#key is the file name, value is the label
        
        
        
        
        #analyze=copy.copy(testx)#this is for analyze the result later
        analyze=copy.copy(testx)#this is for analyze the result later
        
        testx=vector.transform(teststem)
        testpos=vectorpos.transform(testpos)
        testdep=vectordep.transform(testdep)
        
        #testx=sp.hstack((testx, testpos), format='csr')
        testx=sp.hstack((testx, testdep), format='csr')
        
        
        
        TP=0
        FP=0
        FN=0
        TN=0
        
        
        result=clf.predict(testx)
        
        for i in range(len(result)):
            fileName=indexTable.get(i)
            
            if(testy[i]==1):#real positive
                
                if(result[i]>=0.5):#predict positive
                    #print fileName
                    #print i
                    predictTable[fileName]=1
                    TP=TP+1
                if(result[i]<0.5):
                    FN=FN+1
                    print "FN:"
                    print analyze[i]
                    print i
                    
            elif(testy[i]==0):#real negative
                if(result[i]>=0.5):#predict positive
                    #print fileName
                    #print i
                    predictTable[fileName]=1
                    FP=FP+1
                    print "FP:"
                    print analyze[i]
                    print i
                if(result[i]<0.5):
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
        
        
        #wholeDocument(realTable,predictTable)
        
        
        
        
        #while(i<len(testx)):
        #    result=clf.predict(testx[i])
            #if(result>0.5):
                
                
                
            #i++
    
    
    
    
    
    
    '''
    #####
        testx=[]
        testy=[]
        testpos=[]
        testdep=[]
        
        
        testpathDir=os.listdir("data/firefox")
        fillmatrix(subpathDir,testx,testy,testpos,testdep)
        testx=vector.transform(testx).toarray()
        testpos=vector.transform(testpos).toarray()
        testdep=vector.transform(testdep).toarray()
        
        testx=np.concatenate((testx,testpos),axis=1)
        testx=np.concatenate((testx,testdep),axis=1)
        
        i=0
    '''
    '''
        while(i<len(testx)):
            result=clf.predict(testx[i])
            if(result>0.5):
                
                
                
            i++
    '''
    #for(i=0; i<len(testx); i++):
    #    testx[i]
    
    
    
    #for suballDir in testpathDir:
        
    
    
    
    
    
    '''
    X = [[0.70710678, 0., 0.70710678], [ 0., 0.12, 0.], [0.70710678, 0., 0.70710678],[ 0., 0., 0.],]
    y = [5, 0, 5, 0]
   # clf = svm.SVC(kernel = 'linear')
  #  clf.fit(X, y)  
  #  print clf.predict([[0., 0., 0.]])


    cv = CountVectorizer(vocabulary=['hot', 'cold', 'old'])
    data=cv.fit_transform(['pease porridge hot hot hot', 'pease porridge cold', 'pease porridge in the pot', 'nine days old']).toarray()
    
    print data
    
    
    tags = [
    "python tools",
    "linux tools, ubuntu",
    "distributed systems, linux, networking, tools",
    ]
    vect = CountVectorizer(ngram_range=(1,3))#lower bound and upper bound
    tags = vect.fit_transform(tags)
    print(vect.vocabulary_)
    
    print(tags.toarray())
    
    tag2=["tools tools tools ok"]
    
    tag=vect.transform(tag2)
    print(vect.vocabulary_)
    
    print(tag.toarray())
    
    
    a1=np.array([[1,2,3,4,5],[1,2,3,4,5]])
    a2=[[6,7,1,1,1],[0,0,0,0,0]]
    a3=np.concatenate((a1,a2),axis=0)
    
    print(a3)
    '''
    #vectorizer = HashingVectorizer(stop_words = 'english',non_negative = True, n_features = 10000) 
    #fea_train = vectorizer.fit_transform(rows) 
    #X=["asd asd2 fdsf aa.","asd1 dasd dsad.","as1 as2 asd."]
    #vectorizer=CountVectorizer()
    #transformer=TfidfTransformer()
    
    #aa=vectorizer.fit_transform(X)
    
    #tfidf=transformer.fit_transform(vectorizer.fit_transform(X))
    #y = [1, 0, 1,0]
    
    #print aa.toarray()
    
    #clf.fit(aa, y)  
    
    #W=["asd asd2 fdsf"]
    #bb=vectorizer.fit_transform(W)
    
    #print bb.toarray()
    #print clf.predict(aa)
    
    #cc=HashingVectorizer(n_features=7)
    #vector=cc.transform(["I love apple","He hate dog","What do you want","a very big pig"])
    #print vector.toarray()
    #clf = svm.SVC(kernel = 'linear')
    #clf.fit(vector, y)  
    #vector=cc.transform(["pig"])
    #print vector.toarray()
    #print clf.predict(vector)
    
    #vector=cc.transform(["I", "love", "apple","He", "hate", "dog","What", "do", "you", "want","a"])
    #print vector.toarray()
    
    
    '''
        for suballDir in subpathDir:
            #we need to read the xml file
            #title first
            doc = minidom.parse("data/"+allDir+"/"+suballDir)
            title = doc.getElementsByTagName("title")[0]
            titleob = title.getAttribute("ob")            
            x.append(title.firstChild.data)#add the title
            
            titlepos=""
            titledep=""
            for sent in nlp(title.firstChild.data).sents:

                sentnlp=nlp(unicode(str(sent)))
                
                
                for word in sentnlp:
                    titlepos+=" "+word.pos_
                    titledep+=" "+word.dep_
            pos.append(titlepos)#add the pos
            dep.append(titledep)#add the pos
            
            
            if(titleob=="x"):#add the title's lable
                y.append(10)
            else:
                y.append(0)    
            #print("title name: "+title.firstChild.data)
            
            #content second

            
            srs=doc.getElementsByTagName("st")
            for sr in srs:
                ob=sr.getAttribute("ob")
                #print("ob: "+ob)
                content=sr.firstChild.data
                #print("content: "+content)
                x.append(content)
                if(ob=="x"):
                    y.append(10)
                else:
                    y.append(0) 
                    
                    
                contentpos=""
                contentdep=""
                for sent in nlp(title.firstChild.data).sents:
                    sentnlp=nlp(unicode(sent))
                    for word in sentnlp:
                        contentpos+=" "+word.pos_
                        contentdep+=" "+word.dep_
                pos.append(contentpos)#add the pos
                dep.append(contentdep)#add the pos
    '''
    
    
    
    