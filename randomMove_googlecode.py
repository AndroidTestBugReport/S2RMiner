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

os.makedirs('test/google code')

shutil.copytree('random/google500', 'labeled/google code')

time.sleep(0.5)

filenames = random.sample(os.listdir('labeled/google code'), 100)

for file in filenames:
    shutil.move('labeled/google code/'+file, 'test/google code/')
    time.sleep(0.1)
    

print(filenames)
