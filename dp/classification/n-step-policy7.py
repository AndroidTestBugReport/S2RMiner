import os



def findMostStep(comments):
    outputStep=[]
    Max=0
    if len(comments)>0:
        for subList in comments:
            if len(subList)>Max:
                outputStep=subList#policy 6 changed    
    
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
    return outputStep
    
def writeFile(motherFolder,allDir,title,outputStep):
    fileOutput=open(motherFolder+"step-policy7/"+   allDir+"7", 'w')
    
    #for line in outputStep:
    fileOutput.write(title.replace("numDot","").replace("#step","").strip()+os.linesep)###policy 5 change
        
    fileOutput.close()


def policy(motherFolder):
    stepFolder=os.listdir(motherFolder+"step")
    
    
    
    for allDir in stepFolder:
        
        
        title=""
        comments=[]
        
        
        
        fileR = open(motherFolder+"step/"+allDir, 'r')
        contents = fileR.readlines()


        titleTag=False
        commentTag=0
        
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
            
            if titleTag:#this is for the title
                title=line
                title=title.replace("#step","").strip()
                
                
                
            elif commentTag>0:#this is for the comments
                if "####comment#12345#" in line:
                    commentTag+=1
                    comments.append([])
                    continue
                else:                        #line=line.replace("#step","").strip()
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