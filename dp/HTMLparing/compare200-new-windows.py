import sys
import os,shutil
from shutil import copyfile


file = open("compareReport.txt", 'w')
pathDir=os.listdir("compare200")
if not len(pathDir)==2:
    print("you need have exact two folders in the folder compare200")

subFolder1=pathDir[0]
subFolder2=pathDir[1]

subpathDir1=os.listdir("compare200\\"+subFolder1)
subpathDir2=os.listdir("compare200\\"+subFolder2)

if not len(subpathDir1)==len(subpathDir2):
    print("two subfolders should be same size")



countDifferent=0
lineCount=0

puYes_wangYes=0
puYes_wangNo=0

puNo_wangYes=0
puNo_wangNo=0

print("compare start")
file.write("compare start"+os.linesep)





for fileName in subpathDir1:
    sameTag=True
    
    
    try:
        f1 = open("compare200\\"+subFolder1+"\\"+fileName)
        contents1 = f1.readlines()
        
        f2 = open("compare200\\"+subFolder2+"\\"+fileName)
        contents2 = f2.readlines()
    
    except:
        print("file name not same")
        file.write("file name not same"+os.linesep)
        break
    
    
    if not len(contents1)== len(contents2):
        print("file line length different, it should not happened")
        print("file name:"+fileName)
        file.write("file line length different, it should not happened"+os.linesep)
        file.write("file name:"+fileName+os.linesep)
        
        countDifferent=countDifferent+1
        
        copyfile("compare200\\"+subFolder1+"\\"+fileName, "compareResult\\length difference\\"+fileName+"copy1")
        copyfile("compare200\\"+subFolder2+"\\"+fileName, "compareResult\\length difference\\"+fileName+"copy2")
        
        f1.close()
        f2.close()
        
        os.remove("compare200\\"+subFolder1+"\\"+fileName)
        os.remove("compare200\\"+subFolder2+"\\"+fileName)
    
    else:##############length is same
        
        ##########check the illegal ascii
        illegalAscii=False
        for i in range(0,len(contents1)):
            lineCount=lineCount+1
            line1=contents1[i]
            line2=contents2[i]
            
            try:
                line1.encode('ascii',errors='ignore')
                line2.encode('ascii',errors='ignore')
            except:
                countDifferent=countDifferent+1

                print("illegal ascii happens please let the labeler check the wrong special things input")
                copyfile("compare200\\"+subFolder1+"\\"+fileName, "compareResult\\illegal ascii\\"+fileName+"copy1")
                copyfile("compare200\\"+subFolder2+"\\"+fileName, "compareResult\\illegal ascii\\"+fileName+"copy2")
                illegalAscii=True
                break
        if illegalAscii:
            f1.close()
            f2.close()
            os.remove("compare200\\"+subFolder1+"\\"+fileName)
            os.remove("compare200\\"+subFolder2+"\\"+fileName)
            continue
        
        
        
        
        ####here is the normal case
        
        
        newfile=open("compareResult\\Normal\\"+fileName, 'w')
        
        ####check the difference
        for i in range(0,len(contents1)):
            lineCount=lineCount+1
            
            line1=contents1[i]
            line2=contents2[i]
            if not line1 == line2:#not same
                sameTag=False
                print(""+os.linesep)
                print(""+os.linesep)
                print("---------------------------------------")
                print("file different")
                print("1.file name:"+subFolder1+"\\"+fileName)
                print("line string1:"+line1)
                print("2.file name:"+subFolder2+"\\"+fileName)
                print("line string2:"+line2)
                print("---------------------------------------")
                
                
                file.write(""+os.linesep)
                file.write(""+os.linesep)
                file.write("---------------------------------------"+os.linesep)
                file.write("file different"+os.linesep)
                file.write("1.file name:"+subFolder1+"\\"+fileName+os.linesep)
                file.write("line string 1:"+line1+os.linesep)
                file.write("2.file name:"+subFolder2+"\\"+fileName+os.linesep)
                file.write("line string 2:"+line2+os.linesep)
                file.write("---------------------------------------"+os.linesep)
                
                userInput=""
                for i in range(0,1000):
                    try:
                    
                        userInput=input("which one is correct 1 or 2: ")
                        if str(userInput)=="1":
                            newfile.write(line1.encode('ascii',errors='ignore'))
                            break
                        elif str(userInput)=="2":
                            newfile.write(line2.encode('ascii',errors='ignore'))
                            break
                        else:
                            print("you can only input 1 or 2")
                    except:
                        print("you can only input 1 or 2")
                        
                
                
                countDifferent=countDifferent+1
    
            else:#same
                newfile.write(line1.encode('ascii',errors='ignore'))

                
                
        newfile.close()
        f1.close()
        f2.close()
        os.remove("compare200\\"+subFolder1+"\\"+fileName)
        os.remove("compare200\\"+subFolder2+"\\"+fileName)
    
    
    
    
    
    
    
        ####check the kappa
        for i in range(1,len(contents1)):
                line1=contents1[i]
                line2=contents2[i]
                
                
                if not "#####title#12345#" in line1:
                    if not "####comment#12345#" in line1: 
                    
                    
                        if "#step" in line1 or "#wtep" in line1:#Pu yes
                            
                            if "#step" in line2 or "#wtep" in line2:#wang yes
                                puYes_wangYes=puYes_wangYes+1
                            else:#wang no
                                puYes_wangNo=puYes_wangNo+1

                        
                        else:
                            if "#step" in line2 or "#wtep" in line2:#pu no
                                puNo_wangYes=puNo_wangYes+1#wang yes
                            else:
                                puNo_wangNo=puNo_wangNo+1#wang no
                    
    
print("---------------------------------------")
print "how many difference in count:"
print countDifferent
print("---------------------------------------")
                
                
pra=(puYes_wangYes+puNo_wangNo)/float(puYes_wangYes+puNo_wangNo+puYes_wangNo+puNo_wangYes)
print("---------------------------------------")
print "matchrate:"
print pra
print "bothYes:"
print puYes_wangYes
print "bothNo:"
print puNo_wangNo
print "notSame:"
print puYes_wangNo+puNo_wangYes


puYes=(puYes_wangYes+puYes_wangNo)/float(puYes_wangYes+puNo_wangNo+puYes_wangNo+puNo_wangYes)
puNo=(puNo_wangYes+puNo_wangNo)/float(puYes_wangYes+puNo_wangNo+puYes_wangNo+puNo_wangYes)

wangYes=(puYes_wangYes+puNo_wangYes)/float(puYes_wangYes+puNo_wangNo+puYes_wangNo+puNo_wangYes)
wangNo=(puYes_wangNo+puNo_wangNo)/float(puYes_wangYes+puNo_wangNo+puYes_wangNo+puNo_wangYes)

print "puYes:"
print puYes
print "wangYes:"
print wangYes


print "puNo:"
print puNo
print "wangNo:"
print wangNo

pre=puYes*wangYes+puNo*wangNo
Kappa=(pra-pre)/(1-pre)

print "Kappa:"
print Kappa


#print "lineCount:"
#print lineCount

print("---------------------------------------")
                
                
                
                
                
                
                
                
                
                
                
                
                