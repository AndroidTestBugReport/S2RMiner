import os
import random
import shutil
import time

try:
    
    shutil.rmtree('labeled/google code')

except:
    print("aa1")

try:
    

    shutil.rmtree('test/google code')

except:
    print("aa2")
    
try:
    

    shutil.rmtree('labeled/github')
    

except:
    print("aa3")
    
try:
    
    
    shutil.rmtree('test/github')
except:
    print("aa4")


os.makedirs('test/github')

shutil.copytree('random/github500', 'labeled/github')

time.sleep(0.5)

filenames = random.sample(os.listdir('labeled/github'), 100)

for file in filenames:
    shutil.move('labeled/github/'+file, 'test/github/')
    time.sleep(0.1)
    

print(filenames)
