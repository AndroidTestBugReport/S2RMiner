import os
import random
import shutil
import time
import math

targetFolder='google code'

precentAge=20


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
        print countTotal
        

##############move to the test#####################


countMove=precentAge/100.0*countTotal
countMove=math.floor(countMove)
print "countMove"
print countMove
filesToMove=random.sample(os.listdir('buildDataSet/'+targetFolder+"/train"), int(countMove))

for filename in filesToMove:
    shutil.move('buildDataSet/'+targetFolder+"/train/"+filename, 'buildDataSet/'+targetFolder+"/test/"+filename)
    time.sleep(0.01)
