from itertools import permutations 
from statistics import mean 
import numpy as np
from scipy.sparse import vstack
from scipy.sparse import hstack
from scipy import sparse
from scipy.sparse import csr_matrix

#b=sparse.csr_matrix(a)

dataSet=[ ]
file = open("./"+"dataset-result", 'r')
contents = file.readlines()

oneBugreport=[]
for index,item in enumerate(contents):
    
    if (index+1)%11==0:
        oneBugreport.append(item.strip())
        dataSet.append(oneBugreport)
        oneBugreport=[]
    else:
        oneBugreport.append(item.strip())
        
        
#print(dataSet[15])

#indexList=range(0,11)
indexList=range(0,11)
perm = permutations(indexList) 
indexList=list(perm)
print(len(indexList))


runTimeAll=sparse.csr_matrix(np.array([]))
#for i in range(0,len(dataSet)):
for i in range(0,len(dataSet)):
    print(i)
    
    
    runTimeOneReport=[]
    #runTimeOneReport=sparse.csr_matrix(np.array([]))
    
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
        #runTimeAll=vstack([runTimeAll,runTimeOneReport])
        runTimeAll=vstack((runTimeAll,sparse.csr_matrix(np.array(runTimeOneReport))))
    #runTimeAll.append(runTimeOneReport)

print(np.shape(runTimeAll))

#MinIndex=0
#MinValue=1000000


resultList=runTimeAll.mean(0)[0]

minIndex=np.argmin(resultList)
meanResult=np.min(resultList)

'''
#meanList=[]
for i in range(0,len(indexList)):
    #print(i)
    #if i%10000==0:
    print(i)
    
    #meanResult=runTimeAll[:,i].mean()
    #print(runTimeAll[:,i])

    if meanResult< MinValue:
        MinIndex=i
        MinValue=meanResult

'''
print(meanResult)
print(indexList[minIndex])
    

    
    

##673.875
#(2, 10, 0, 1, 3, 9, 4, 6, 5, 7, 8)




#print(len(list(perm)))