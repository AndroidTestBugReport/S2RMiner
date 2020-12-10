from itertools import permutations 
from statistics import mean 
import numpy as np
from scipy.sparse import vstack
from scipy import sparse
from scipy.sparse import csr_matrix
        
#print(dataSet[15])
class TreeNode(object):
     def __init__(self, x):
         self.val = x
         self.children =[]#by node
         self.parent = []#by index

def buildTree(policyList,node):
    parentList=node.parent
    
    print(node.parent)
    
    for index in range(0,11):
        if not index in parentList:
            if "N" in policyList[index]:
                newnode=TreeNode(int(policyList[index].replace("N",""))+node.val)
                newnode.parent=parentList.append(index)
                node.children.append(newnode)
                buildTree(policyList,newnode)       
                
            else:
                newnode=TreeNode(int(policyList[index])+node.val)
                newnode.parent=parentList.append(index)
                
                
    
    

#indexList=range(0,11)    

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



p=TreeNode(None)
for i in range(0,len(dataSet)):
    print(i)
    node=TreeNode(0)
    buildTree(dataSet[i],node)









    
    





#print(len(list(perm)))