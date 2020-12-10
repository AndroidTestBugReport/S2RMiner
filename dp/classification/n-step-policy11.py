import os



def findMostStep(comments):
    outputStep=[]
    '''
    Max=0
    for subList in comments:
        count=0
        for line in subList:
            if "#step" in line:
                count+=1
        
        if count>Max:
            outputStep=subList
            Max=len(subList)
    '''
    if len(comments)>0:
        outputStep=comments[0]
    return outputStep
    
def writeFile(motherFolder,allDir,title,outputStep):
    fileOutput=open(motherFolder+"step-policy11/"+   allDir+"11", 'w')
    
    for line in outputStep:
            fileOutput.write(line.replace("#step","").replace("numDot","").strip()+os.linesep)
    fileOutput.close()


def policy(motherFolder):
    stepFolder=os.listdir(motherFolder+"step")
    
    
    
    for allDir in stepFolder:
        
        
        title=""
        comments=[]
        
        
        
        fileR = open(motherFolder+"step/"+allDir, 'r')
        contents = fileR.readlines()
        print(allDir)

        titleTag=False
        commentTag=0
        
        toReproduceIndex=-1
        
        for i in range(0,len(contents)-1):
            line=contents[i]
            
            
            
            if "#####title#12345#" in line:
                if i==0:
                    titleTag=True#this is the title
                    continue
                else:
                    titleTag=False
                    commentTag=1#it means the first sentence of a somment
                    comments.append([])
                    continue
            
            if titleTag==True and "#step" in line:#this is for the title
                title=line
                title=title.replace("#step","").strip()
                
                
                
            elif commentTag>0:#this is for the comments
                
                
                
                if "to reproduce" in line.lower():
                    toReproduceIndex=i
                    continue
            
                if toReproduceIndex==-1:
                    continue
                
                
                
                
                
                
                
                if "####comment#12345#" in line:
                    commentTag+=1
                    comments.append([])
                    toReproduceIndex=-1
                    continue
                else:
                    ###policy changed
                    ####### this one moved from the CNN-LSTm part
                    item1=""#2 before sentence
                    item2=""#2 behind sentence
                    item3=""#1 before sentence
                    item4=""#1 behind sentence
                    item5=""#current sentence
                    
                    
                    
                    #########fill before sentences
                    lineBefore=3
                    
                    
                    for kk in range(1,3):#[1,2] check where is the comment break, record as lineBefore
                        if "####comment#12345#" in contents[i-kk] or "#####title#12345#" in contents[i-kk]:
                           lineBefore=kk
                           break
                    
                    if lineBefore==3:
                        item1=contents[i-2]
                        item3=contents[i-1]
                        
                
                    if lineBefore==2:
                        item1="####comment#12345#"
                        item3=contents[i-1]
                        
                        
                    if lineBefore==1:
                        item1="####comment#12345#"
                        item3="####comment#12345#"
                        
                           
                    ##########fill behind sentences
                    lineBehind=3
                    
                    for kk in range(1,3):#[1,2] check where is the comment break, record as lineBefore
                        if "####comment#12345#" in contents[i+kk] or "#####title#12345#" in contents[i+kk]:
                           lineBehind=kk
                           break
                    
                    
                    if lineBehind==3:
                        item2=contents[i+2]
                        item4=contents[i+1]
                
                    
                    if lineBehind==2:
                        item2="####comment#12345#"
                        item4=contents[i+1]
                    
                    if lineBehind==1:
                        item2="####comment#12345#"
                        item4="####comment#12345#"
                    
                    
                    
                    if "#step" in item1 or "#step" in item2 or "#step" in item3 or "#step" in item4 or "#step" in item5:
                        #line=line.strip()
                        comments[commentTag-1].append(line)
                        print(line)                

        '''
        if "carreport" in allDir:
            print("bingo")
        '''
        outputStep=findMostStep(comments)        
        #print(comments)
        writeFile(motherFolder,allDir,title,outputStep)
        
    
    
    

if __name__ == '__main__':
    
    
    
    motherFolder="recdroid-labeled-by-model/"
    
    policy(motherFolder)