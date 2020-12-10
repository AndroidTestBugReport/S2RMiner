import sys
import os,shutil
import math



for filename in os.listdir("recdroid-labeled-by-model/oracle"):
    f1=open("recdroid-labeled-by-model/oracle/"+filename)
    contents=f1.readlines()
    crashTag=False
    for line in contents:
        if "#oracle" in line:
            crashTag=True
            break
        
    if not crashTag:
        print(filename)


                
                
                
                
                