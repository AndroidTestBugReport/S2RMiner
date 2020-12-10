import sys
import os,shutil
from shutil import copyfile


def removeTag(line):
    lineNew=line.replace("#step","")
    lineNew=lineNew.replace("#wtep","")
    lineNew=lineNew.replace("#oracle","")
    lineNew=lineNew.replace("numDot","")
    return lineNew.strip()


file = open("./"+"compareReport.txt", 'w')

subFolder1="new"
subFolder2="old"

subpathDir1=os.listdir("transferLabels-all/"+subFolder1)
subpathDir2=os.listdir("transferLabels-all/"+subFolder2)

if not len(subpathDir1)==len(subpathDir2):
    print("two subfolders should be same size")




for folder in subpathDir2:

    for fileName in os.listdir("transferLabels-all/"+subFolder2+"/"+folder):
        
        
        
        contentsNew=[]
        try:
            f1 = open("transferLabels-all/"+subFolder1+"/"+fileName)
            contents1 = f1.readlines()
            
            f2 = open("transferLabels-all/"+subFolder2+"/"+folder+"/"+fileName)
            contents2 = f2.readlines()
        
        except:
            print("file name not same")
            file.write("file name not same"+os.linesep)
            break
        
        if not len(contents1)== len(contents2):#########3check the length same
            print("transferLabels-all/"+subFolder2+"/"+folder+"/"+fileName)

            j=0
            for i in range(0,len(contents1)):
                
                step=False
                wtep=False
                crash=False
                
                if "Memento" in fileName:
                    print("bingo")
                
                
                if "Bug in app calendar which leads" in contents1[i]:
                    print("bingo")
                
                if contents1[i].strip() == removeTag(contents2[j]): ############3totally same line
                    if "#step" in contents2[j]:
                        step=True
                    if "#wtep" in contents2[j]:
                        wtep=True
                    if "#oracle" in contents2[j]:
                        crash=True 
                    
                    if step:
                        contents1[i]=contents1[i].strip()+" #step"
                    if wtep:
                        contents1[i]=contents1[i].strip()+" #wtep"
                    if crash:
                        contents1[i]=contents1[i].strip()+" #oracle"
                    contentsNew.append(contents1[i].strip()+os.linesep)
                    
                
                else:
                    while(True):
                        if "#step" in contents2[j]:
                            step=True
                        if "#wtep" in contents2[j]:
                            wtep=True
                        if "#oracle" in contents2[j]:
                            crash=True 
                        
                        if  removeTag(contents2[j+1]) in contents1[i].strip():
                            j=j+1
                            
                        else:
                            if step:
                                contents1[i]=contents1[i].strip()+" #step"
                            if wtep:
                                contents1[i]=contents1[i].strip()+" #wtep"
                            if crash:
                                contents1[i]=contents1[i].strip()+" #oracle"
                            contentsNew.append(contents1[i].strip()+os.linesep)
                            break
                
                j=j+1
                    
                    
            f1.close()
            f2.close()
            
            f3= open("transferLabels-all/"+"result"+"/"+folder+"/"+fileName, 'w')
            
            for line in contentsNew:
                f3.write(line)
            f3.close()
                
        else:
            copyfile("transferLabels-all/"+subFolder2+"/"+folder+"/"+fileName, "transferLabels-all/"+"result"+"/"+folder+"/"+fileName)       
                
                
                
                
                
                
                
                