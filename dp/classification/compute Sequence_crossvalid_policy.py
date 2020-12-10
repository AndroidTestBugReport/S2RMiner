from itertools import permutations 
from statistics import mean 
import numpy as np
from scipy.sparse import vstack
from scipy.sparse import hstack
from scipy import sparse
from scipy.sparse import csr_matrix
import numpy as np
from sklearn.model_selection import KFold,StratifiedKFold
import random



def computeOptimal(dataSet,indexList):
    runTimeAll=sparse.csr_matrix(np.array([]))
    for i in range(0,len(dataSet)):
        runTimeOneReport=[]
        #runTimeOneReport=sparse.csr_matrix(np.array([]))
        print(i)
        #cou=0
        for oneIndexSeq in indexList:
            #cou+=1
            #print(cou)
            
            time=0
            for item in oneIndexSeq:
                if "N" in dataSet[i][item]:
                    time=time+int(dataSet[i][item].replace("N",""))
                else:
                    time=time+int(dataSet[i][item])
                    runTimeOneReport.append(time)
                    #runTimeOneReport=hstack([runTimeOneReport,time])
                    
                    break
        #print(len(runTimeOneReport))
        
        
        if i==0:
            #runTimeAll=runTimeOneReport
            runTimeAll=sparse.csr_matrix(np.array(runTimeOneReport))
            #runTimeAll=np.array([runTimeOneReport])
        else:
            #runTimeAll=vstack((runTimeAll,sparse.csr_matrix(np.array(runTimeOneReport))))
            #runTimeAll.sum(0)
            runTimeAll=runTimeAll+sparse.csr_matrix(np.array(runTimeOneReport))
    
    
    print(np.shape(runTimeAll))

    #MinIndex=0
    #MinValue=1000000
    
    resultList=(runTimeAll/len(dataSet)).toarray()
    #resultList=runTimeAll.mean(0)[0]
    
    minIndex=np.argmin(resultList)
    meanResult=np.min(resultList)
    print(meanResult)
    print(indexList[minIndex])
    
    return indexList[minIndex]



def computTimeDifferent(testDataSet,trainIndexList,testIndexList):
    
    
    
    trainedTime=0
    testOptimalTime=0
    
    
    for i in range(0,len(testDataSet)):
        
        
        for item in trainIndexList:
            if "N" in testDataSet[i][item]:
                trainedTime=trainedTime+int(testDataSet[i][item].replace("N",""))
            else:
                trainedTime=trainedTime+int(testDataSet[i][item])                
                break
        
        
    
    
    for i in range(0,len(testDataSet)):        
        
        for item in testIndexList:
            if "N" in testDataSet[i][item]:
                testOptimalTime=testOptimalTime+int(testDataSet[i][item].replace("N",""))
            else:
                testOptimalTime=testOptimalTime+int(testDataSet[i][item])                
                break    
        
    
    
    return  trainedTime/len(testDataSet),testOptimalTime/len(testDataSet)



def computeFoldSet(dataSetWhole,trainIndex):
    
    trainDataSet=[]
    for index in trainIndex:
        trainDataSet.append(dataSetWhole[index])
    
    
    return trainDataSet
    
    
#b=sparse.csr_matrix(a)

dataSetWhole=[ ]
file = open("./"+"dataset-result", 'r')
contents = file.readlines()

oneBugreport=[]
for index,item in enumerate(contents):
    
    if (index+1)%11==0:
        oneBugreport.append(item.strip())
        dataSetWhole.append(oneBugreport)
        oneBugreport=[]
    else:
        oneBugreport.append(item.strip())
        
        
#print(dataSet[15])

#indexList=range(0,11)
indexList=range(0,11)
perm = permutations(indexList) 
indexList=list(perm)
print(len(indexList))


#for i in range(0,len(dataSet)):


#################
#dataSetWhole=dataSetWhole[1:6]

y=np.array(np.zeros(len(dataSetWhole)))



random.shuffle(dataSetWhole)
random.shuffle(dataSetWhole)
random.shuffle(dataSetWhole)

folder = KFold(n_splits=4,random_state=0,shuffle=False)
#folder = StratifiedKFold(n_splits=4,random_state=0,shuffle=False)


for trainIndex, testIndex in folder.split(dataSetWhole,y):
    
    
    #trainDataSet=dataSetWhole[trainIndex]
    
    print("train")
    trainDataSet=computeFoldSet(dataSetWhole,trainIndex)
    trainIndexList=computeOptimal(trainDataSet,indexList)
    #print(trainIndexList)
    
    print("test")
    testDataSet=computeFoldSet(dataSetWhole,testIndex)
    testIndexList=computeOptimal(testDataSet,indexList)
    #print(testIndexList)
    
    trainedTime,testOptimalTime=computTimeDifferent(testDataSet,trainIndexList,testIndexList)
    trainPercentage=float(trainedTime)/testOptimalTime-1####trained=(1+x%)opt   how many percentage increased
    
    
    
    
    
    randomList=random.choice(indexList)
    randomTime,testOptimalTime=computTimeDifferent(testDataSet,randomList,testIndexList)
    randomPercentage=float(randomTime)/testOptimalTime-1####trained=(1+x%)opt   how many percentage increased
    
    print("random")
    print(randomList)
    
    
    
    
    outputFile = open("policy-cross-result/3fold.txt", 'a')
    outputFile.write("\n")
    outputFile.write("trained\n")
    outputFile.write("trainIndexList:  "+str(trainIndexList) +"\n")
    outputFile.write("testIndexList:  "+str(testIndexList) +"\n")
    outputFile.write("trainedTime:  "+str(trainedTime) +"\n")
    outputFile.write("testOptimalTime:  "+str(testOptimalTime) +"\n")
    outputFile.write("trainPercentage:  "+str(trainPercentage) +"\n")
    outputFile.write("randomTime:  "+str(randomTime) +"\n")
    outputFile.write("randomIndexList:"+str(randomList)+"\n")
    outputFile.write("randomPercentage:  "+str(randomPercentage) +"\n")
    
    outputFile.close()
    
    
    
    
    

    
    

##673.875
#(2, 10, 0, 1, 3, 9, 4, 6, 5, 7, 8)




#print(len(list(perm)))