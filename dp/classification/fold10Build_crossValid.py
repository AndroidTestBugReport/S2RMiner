import os
import random
import shutil
import time
import math

#targetFolder='google code'

precentAge=0


try:
    shutil.rmtree('cross-valid'+'/test')
except:
    print("fail to remove test")
    
    
try:
    shutil.rmtree('cross-valid'+'/train')
except:
    print("fail to remove train")
    
    

os.makedirs('cross-valid'+'/train')
os.makedirs('cross-valid'+'/test')
    
    
time.sleep(0.5)
originalFolderName='cross-valid'+'/originalFolders'
pathDir=os.listdir(originalFolderName)


countTotal=0
for subpath in pathDir:
    subpathDir=os.listdir(originalFolderName+'/'+subpath)
    for filename in subpathDir:
        shutil.copyfile(originalFolderName+"/"+subpath+"/"+filename,'cross-valid'+"/train/"+filename)
        time.sleep(0.02)
        countTotal=countTotal+1
        print(countTotal)
        

##############move to the test#####################


countMove=precentAge/100.0*countTotal
countMove=math.floor(countMove)
print("countMove")
print(countMove)
filesToMove=random.sample(os.listdir('cross-valid'+"/train"), int(countMove))

for filename in filesToMove:
    shutil.move('cross-valid'+"/train/"+filename, 'cross-valid'+"/test/"+filename)
    time.sleep(0.02)
