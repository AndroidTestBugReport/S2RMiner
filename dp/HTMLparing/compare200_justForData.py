import sys
import os,shutil



file = open("./"+"compareReport.txt", 'w')
pathDir=os.listdir("compare200")
if not len(pathDir)==2:
    print("you need have exact two folders in the folder compare200")

subFolder1=pathDir[0]
subFolder2=pathDir[1]

subpathDir1=os.listdir("compare200/"+subFolder1)
subpathDir2=os.listdir("compare200/"+subFolder2)

if not len(subpathDir1)==len(subpathDir2):
    print("two subfolders should be same size")



countDifferent=0
lineCount=0

puYes_wangYes=0
puYes_wangNo=0

puNo_wangYes=0
puNo_wangNo=0


OpuYes_wangYes=0
OpuYes_wangNo=0

OpuNo_wangYes=0
OpuNo_wangNo=0






print("compare start")
file.write("compare start"+os.linesep)

for fileName in subpathDir1:
    sameTag=True
    
    
    try:
        f1 = open("compare200/"+subFolder1+"/"+fileName)
        contents1 = f1.readlines()
        
        f2 = open("compare200/"+subFolder2+"/"+fileName)
        contents2 = f2.readlines()
    
    except:
        print("file name not same")
        file.write("file name not same"+os.linesep)
    
    
    if not len(contents1)== len(contents2):
        print("file line length different, it should not happened")
        print("file name:"+fileName)
        file.write("file line length different, it should not happened"+os.linesep)
        file.write("file name:"+fileName+os.linesep)
        
        
        countDifferent=countDifferent+1
    
    else:
        ####check the difference
        for i in range(1,len(contents1)):
            lineCount=lineCount+1
            
            line1=contents1[i]
            line2=contents2[i]
            
            
            
            
            if not line1.strip() == line2.strip():
                sameTag=False
                print(""+os.linesep)
                print(""+os.linesep)
                print("---------------------------------------")
                print("file different")
                print("file name:"+subFolder1+"/"+fileName)
                print("line string1:"+line1)
                print("file name:"+subFolder2+"/"+fileName)
                print("line string2:"+line2)
                print("---------------------------------------")
                
                
                file.write(""+os.linesep)
                file.write(""+os.linesep)
                file.write("---------------------------------------"+os.linesep)
                file.write("file different"+os.linesep)
                file.write("file name:"+subFolder1+"/"+fileName+os.linesep)
                file.write("line string1:"+line1+os.linesep)
                file.write("file name:"+subFolder2+"/"+fileName+os.linesep)
                file.write("line string2:"+line2+os.linesep)
                file.write("---------------------------------------"+os.linesep)
                
                
                countDifferent=countDifferent+1
    
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
                                
                                
                                
                                
                if not "#####title#12345#" in line1:
                    if not "####comment#12345#" in line1: 
                    
                    
                        if "#oracle" in line1:#Pu yes
                            
                            if "#oracle" in line2:#wang yes
                                OpuYes_wangYes=OpuYes_wangYes+1
                            else:#wang no
                                OpuYes_wangNo=OpuYes_wangNo+1

                        
                        else:
                            if "#oracle" in line2:#pu no
                                OpuNo_wangYes=OpuNo_wangYes+1#wang yes
                            else:
                                OpuNo_wangNo=OpuNo_wangNo+1#wang no
                                
                                
                                
        
                    
    
print("---------------------------------------")
print "how many difference in count:"
print countDifferent
print("---------------------------------------")
                
                
pra=(puYes_wangYes+puNo_wangNo)/float(puYes_wangYes+puNo_wangNo+puYes_wangNo+puNo_wangYes)
print("---step:------------------------------------")
#print "matchrate:"
#print pra
print "bothYes:"
print puYes_wangYes
print "bothNo:"
print puNo_wangNo
print "notSame:"
print puYes_wangNo+puNo_wangYes


print("---oracle:------------------------------------")
#print "matchrate:"
#print pra
print "bothYes:"
print OpuYes_wangYes
print "bothNo:"
print OpuNo_wangNo
print "notSame:"
print OpuYes_wangNo+OpuNo_wangYes






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
                
                
                
                
                
                
                
                
                
                
                
                
                