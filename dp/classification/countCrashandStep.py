import sys
import os,shutil

matchStep=["#step","#wtep"]
match=["#oracle","#oracle"]

#match=["#oracle"]
tableclick = {'click':0, 'choose':0, 'select':0, 'launch':0, 'pick':0, 'tap':0, 'open':0, 'press':0, 'go':0, 'select':0}
tableinput = {'input':0, 'enter':0, 'type':0, 'insert':0, 'fill':0, 'change':0, 'write':0, 'set':0, 'put':0, 'add':0}
tableback = {'ok': 0, 'cancel':0, 'done':0, 'back': 0, 'zoom': 0, 'swipe':0, 'rotat':0}
tablespecial={'apostrophe':0 ,'comma' :0, 'colon' :0, 'semicolon' :0, 'hyphen' :0, 'parentheses' :0, 'quote' :0, 'space' :0}
tablespecial2={'"\'':0 ,'",' :0, '":' :0, '";' :0, '"-' :0, '"(' :0, '"\\"' :0, '" "' :0}



trainFolder='buildDataSet/google code/originalFolders'

trainPathDir=os.listdir(trainFolder)

totalOracleCount=0
stepInOracleCount=0
rotateCount=0
Min=1000000
Max=0

TotalStep=9

fileCount=0

inputCount=0
specialAccount=0

okC=0
doneC=0
cancelC=0

genericCount=0

for folder in trainPathDir:
    fileDir=os.listdir(trainFolder+"/"+folder)
    
    for fileName in fileDir:
        fileCount+=1
        file = open(trainFolder+"/"+folder+"/"+fileName, 'r')
        contents = file.readlines()
        oracleTag=False
        for line in contents:
            if match[0] in line or match[1] in line:
                totalOracleCount+=1
                oracleTag=True
                break
        
        stepTag=False
        if oracleTag:
            for line in contents:
                if matchStep[0] in line or matchStep[1] in line:
                    stepInOracleCount+=1
                    stepTag=True
                    break
        
        if stepTag:
            stepNum=0
            for line in contents:
                if matchStep[0] in line or matchStep[1] in line:
                    stepNum+=1
            if stepNum>Max:
                Max=stepNum
            if stepNum<Min:
                Min=stepNum
            TotalStep+=stepNum
            
        if oracleTag:
            for line in contents:
                if "rotat" in line:
                    #print(line)
                    #print(trainFolder+"/"+folder+"/"+fileName)
                    rotateCount+=1
                    break
            
        if stepTag:
            for line in contents:
                intPutTag=False
                for actionWord in tableinput.keys():
                    if actionWord in line:
                        inputCount+=1
                        intPutTag=True
                        break
                if intPutTag:
                    break
        if stepTag:
            for line in contents:
                specialTag=False
                
                for actionWord in tablespecial.keys():
                    if actionWord in line:
                        specialAccount+=1
                        specialTag=True
                        break
                
                '''
                if not specialTag:
                    for actionWord in tablespecial2.keys():
                        if actionWord in line:
                            specialAccount+=1
                            specialTag=True
                            break
                '''
                if specialTag:
                    break
        
        
        if stepTag:
            ok=False
            done=False
            canncel=False
            
            
            for line in contents:
                if not ok and "ok" in line and "tep" in line:
                    print(trainFolder+"/"+folder+"/"+fileName)
                    okC+=1
                    ok=True
                    
                if not done and "done" in line and "tep" in line:
                    tableback["done"]+=1
                    doneC+=1
                    done=True
                    
                if not canncel and "canncel" in line and "tep" in line:
                    tableback["canncel"]+=1
                    cancelC+=1
                    canncel=True
                
            if ok or done or canncel:
                genericCount+=1

        

print("file number")
print(fileCount)
print("oracle number")
print(totalOracleCount)
print("step in oracle number")
print(stepInOracleCount)
print("Max")
print(Max)
print("MIn")
print(Min)
print("Average")
print(TotalStep/stepInOracleCount)
print("Rotate count")
print(5)
print("inputCount")
print(inputCount)
print("specialCount")
print(specialAccount)
print("ok")
print(okC)
print("done")
print(doneC)
print("cancel")
print(cancelC)
print("genericcount")
print(genericCount)


