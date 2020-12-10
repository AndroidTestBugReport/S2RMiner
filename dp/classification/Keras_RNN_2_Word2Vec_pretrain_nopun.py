import sys
import os,shutil
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import re
from gensim.test.utils import get_tmpfile
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
import string




def sen2List(pathDir, wordsDocList):
    for fileName in pathDir:
        f = open(trainFolder+"/"+fileName)
        contents = f.readlines()
        for oneLine in contents:
        
            oneLine=oneLine.replace("#step","")
            oneLine=oneLine.replace("#wtep","")
            oneLine=oneLine.replace("#oracle","")
            oneLine=oneLine.strip()
            oneLine = oneLine.lower().translate(table)
            wordsSentList=oneLine.split()
            #wordList=oneLine.split(" ")###this one is with punctuation
            #wordsSentList=re.findall(r"[\w']+|[.,!?;]", oneLine.lower())####this one is without punctuation
            #print wordsSentList
            wordsDocList.append(wordsSentList)
        







if __name__ == '__main__':
    
    table = str.maketrans({key: None for key in string.punctuation})
    '''
    path = get_tmpfile("word2vec.model")
    model = Word2Vec(common_texts, size=100, window=5, min_count=1, workers=4)
    model.save("word2vec.model")
    '''
    vectorsize=50
    
    wordsDocList=[]
    targetFolder='all'
    trainFolder='buildDataSet/'+targetFolder+'/google code'
    pathDir=os.listdir(trainFolder)
    
    sen2List(pathDir, wordsDocList)
    trainFolder='buildDataSet/'+targetFolder+'/github'
    pathDir=os.listdir(trainFolder)
    
    sen2List(pathDir, wordsDocList)
    
    
    ##############3 this is the API of doc2vec
    #documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(wordsDocList)]
    
    model = Word2Vec(wordsDocList,size=vectorsize,window=10,min_count=0,workers=10,iter=10)
    model.save("my_word2vec_model_"+str(vectorsize)+"_noPun")
    '''
    word_vectors = model.wv
    if 'word' in word_vectors.vocab:
        print("bingo")
    
    print("not")
    '''
    '''
    w1=["click"]
    aa=model.wv.most_similar(positive=w1, topn=6)
    print(aa)
    
    vector = model.wv['computer']
    print(vector)
    '''
    '''
    model = Doc2Vec(documents, vector_size=200, window=10, min_count=1, workers=4)
    
    #fname = get_tmpfile("my_doc2vec_model")
    fname="./my_doc2vec_model"
    model.save(fname)
    
    
    model1 = Doc2Vec.load(fname)
    vector = model1.infer_vector(["I", "like", "this"])
    print(vector)
    '''
            
            


            
            
            
            
            
    

    

