import sys
import os,shutil


trackerName="google code"
pathDir=os.listdir("divide100/SentenceSegmentTrain/"+trackerName)

filesNmuberInFloder=200
folderID=0



i=filesNmuberInFloder+1
currentFolder=""

for allDir in pathDir:
    if i>filesNmuberInFloder:
        folderID=folderID+1
        i=1
        currentFolder="divide100/"+trackerName+"/ID"+str(folderID)
        os.mkdir(currentFolder)
    shutil.copyfile("divide100/SentenceSegmentTrain/"+trackerName+"/"+allDir,currentFolder+"/"+allDir)
    
            
    i=i+1

