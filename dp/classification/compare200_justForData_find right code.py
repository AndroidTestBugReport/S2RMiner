import sys
import os,shutil
import math




file = open("./"+"compareReport.txt", 'w')



pathDir=os.listdir("recdroid-txt")


for subfolder in pathDir:
    for filename in os.listdir("recdroid-txt/"+subfolder):
        f1 = open("recdroid-txt-2/"+subfolder+"/"+filename)###ole
        contents1 = f1.readlines()
        
        f2 = open("recdroid-txt/"+subfolder+"/"+filename)
        contents2 = f2.readlines()
        
        if not len(contents1)== len(contents2):
            print("line different")
            print(filename)
        else:
            for i in range(1,len(contents1)):
                newline=contents1[i]
                newline=newline.replace("*","")
                newline=newline.replace("#oracle","")
                newline=newline.replace("#step","")
                newline=newline.replace("#wtep","")
                
                if not newline.strip()==contents2[i].strip():
                    print("content difference")
                    print(filename)


                
                
                
                
                