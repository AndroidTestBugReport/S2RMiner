import sys
import os,shutil
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import re
from gensim.test.utils import get_tmpfile




def sen2List(pathDir, wordsDocList):
    for fileName in pathDir:
        f = open(trainFolder+"/"+fileName)
        contents = f.readlines()
        for oneLine in contents:
        
            oneLine=oneLine.replace("#step","")
            oneLine=oneLine.replace("#wtep","")
            oneLine=oneLine.replace("#oracle","")
            oneLine=oneLine.strip()
            #wordList=oneLine.split(" ")###this one is with punctuation
            wordsSentList=re.findall(r"[\w']+|[.,!?;]", oneLine)####this one is without punctuation
            #print wordsSentList
            wordsDocList.append(wordsSentList)
        




if __name__ == '__main__':
    
    
    
    targetFolder='google code'
    trainFolder='buildDataSet/'+targetFolder+'/train'
    pathDir=os.listdir(trainFolder)
    
    wordsDocList=[]
    sen2List(pathDir, wordsDocList)
    
    ##############3 this is the API of doc2vec
    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(wordsDocList)]
    model = Doc2Vec(documents, vector_size=200, window=10, min_count=1, workers=4)
    
    #fname = get_tmpfile("my_doc2vec_model")
    fname="./my_doc2vec_model"
    model.save(fname)
    
    
    model1 = Doc2Vec.load(fname)
    vector = model1.infer_vector(["I", "like", "this"])
    print vector
    
            
            


            
            
            
            
            
    

    

