import os
import sys
from xml.dom.minidom import Document
from xml.dom import minidom
from lxml import html
#from lxml import etree
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
from lxml import etree
import re

def divdieAndWrite(line,tagLi,file, tokenizer):
    
    
    try:
        tokenizer.train(line)
    except:
        print("error segment")
        
    for sentence in tokenizer.tokenize(line):
        
        if tagLi:
        
            try:
                lineList=sentence.split("\n")
                for line in lineList:
                    if not line.strip()=="":
                        file.write("numDot "+line.strip()+os.linesep)       
                    
            except:    
                print("error symbol")
            #print sentence.strip()
            
            
        else:
            
            try:
                lineList=sentence.split("\n")
                for line in lineList:
                    if not line.strip()=="":
                        file.write(line.strip()+os.linesep)
            except:    
                print("error symbol")
            #print sentence.strip()

def TrackerFolders(subpathDir, tokenizer):
    
    
    
        for suballDir in subpathDir:
            print(suballDir)

            file = open("recdroid-txt/"+allDir+"/"+suballDir[:-5]+".txt", 'w')
            
            liSet=set([])
            liList=[]
            pSet=set([])
            
            
            '''
            ###test
            tree = html.parse("/home/yu/Downloads/labeledtest_checked/labeledtest_checked/18.html")
            #buyers = tree.xpath('//markdown-widget[@text="comment.content"]/.//text()')
            #print("aa")
            buyers=tree.xpath('//markdown-widget[@text="comment.content"]')
            
            for buyer in buyers:#one buyer is one comment
                lineList=buyer.xpath('.//text()')
                for lineText in lineList:
                    
                    removeNewLine=lineText.replace("\n","")
                    
                    print removeNewLine
            
            print("aaa")
            '''
            treeold = html.parse("recdroid-dataset/"+allDir+"/"+suballDir)
            htmlString=etree.tostring(treeold)
            
            clean0 =re.compile('<a.*?>')
            clean1 =re.compile('<a.*?>')
            clean2 =re.compile('<a.*?>')
            clean3 =re.compile('<a.*?>')

            
            
            tree=etree.fromstring(htmlString)
            
            if allDir=="google code":
            #if True:
                #tree = html.parse("recdroid-dataset/"+allDir+"/"+suballDir)
                                ######every comments
                buyers=tree.xpath('//markdown-widget[@text="comment.content"]')
                ######li part
                liTag=tree.xpath('//markdown-widget[@text="comment.content"]/.//li')
                ######title part
                titleXpath=tree.xpath('//div[@id="gca-project-header"]/p[@class="ng-binding"]')[0]
                ######p part
                pAlert=tree.xpath('//markdown-widget[@text="comment.content"]/.//p/a')# sometimes, the xpath  .//text() may divide a p into different p. it is not correct. We need to send alert.
                ######li herf
                liAlert=tree.xpath('//markdown-widget[@text="comment.content"]/.//li/a')
                ######li Alert
                liAlertCode=tree.xpath('//markdown-widget[@text="comment.content"]/.//li/code')
                ######strong part
                liAlertStrong=tree.xpath('//markdown-widget[@text="comment.content"]/.//li/strong')
                ######p strong part
                pAlertStrong=tree.xpath('//markdown-widget[@text="comment.content"]/.//p/a/strong')
                ######p code part
                pAlertCode=tree.xpath('//markdown-widget[@text="comment.content"]/.//p/a/code')
                ######p em part
                pAlertEm=tree.xpath('//markdown-widget[@text="comment.content"]/.//p/a/em')
                ######li em part
                liAlertEm=tree.xpath('//markdown-widget[@text="comment.content"]/.//li/a/em')
                
                
                
                                

            elif allDir=="github":
                
                #tree = html.parse("recdroid-dataset/"+allDir+"/"+suballDir)
                ######every comments                
                buyers=tree.xpath('//td')
                ######li part
                liTag=tree.xpath('//td/.//li')
                ######title part
                titleXpath=tree.xpath('//span[@class="js-issue-title"]')[0]
                ######p part
                pAlert=tree.xpath('//td/.//p/a')
                ######li part
                liAlert=tree.xpath('//td/.//li/a')
                ######code part
                liAlertCode=tree.xpath('//td/.//li/code')
                ######strong part
                liAlertStrong=tree.xpath('//td/.//li/strong')
                ######p strong part
                pAlertStrong=tree.xpath('//td/.//p/strong')
                ######p code part
                pAlertCode=tree.xpath('//td/.//p/code')
                ######p em part
                pAlertEm=tree.xpath('//td/.//p/em')
                ######li em part
                liAlertEm=tree.xpath('//td/.//li/em')
                
            
            else:
                continue
            
            
            
            
            ##################p alert
            addPset=[]
            addPset.append(pAlert)
            addPset.append(liAlert)
            addPset.append(liAlertCode)
            addPset.append(liAlertStrong)
            addPset.append(pAlertStrong)
            addPset.append(pAlertCode)
            addPset.append(pAlertEm)
            addPset.append(liAlertEm)
            
            for addItem in addPset:
                for pItem in addItem:
                    if not len(pItem.xpath('.//text()'))==0:
                        pText=pItem.xpath('.//text()')[0]
                        pSet.add(pText)# add the sensitive word
            '''
            for pItem in pAlert:
                if not len(pItem.xpath('.//text()'))==0:
                    pText=pItem.xpath('.//text()')[0]
                    pSet.add(pText)# add the sensitive word
            
            for liItem in liAlert:####take all the alert to the pset to combine
                if not len(liItem.xpath('.//text()'))==0:
                    liText=liItem.xpath('.//text()')[0]
                    pSet.add(liText)
                    
            for liItem in liAlertStrong:####take all the alert to the pset to combine
                if not len(liItem.xpath('.//text()'))==0:
                    liText=liItem.xpath('.//text()')[0]
                    pSet.add(liText)
            '''          
            
            ##################take title text
            if not len(titleXpath.xpath('.//text()'))==0:
                titleline=titleXpath.xpath('.//text()')[0]#title only have one line
            else:
                titleline=""
                
            
            titleText=titleline.replace("\n"," ")#change <p>'s \n to a space
            titleText=titleline.replace("  "," ")#change two space to one space. The two space may be added by above step by mistake
            
            ##################take li text
            for liItem in liTag:
                liTextList=liItem.xpath('.//text()')
                for liText in liTextList:
                    liText=liText.replace("\n"," ")
                    liText=liText.replace("  "," ")
                    if not len(liText.split())==0: #check whether it is null line         
                        liSet.add(liText)#here is the set 
                        liList.append(liText)
                
                
            
            
            
            ##############this is for the title
            file.write("#####title#12345#"+os.linesep)
            
            divdieAndWrite(titleText,False,file, tokenizer)

            file.write("#####title#12345#"+os.linesep)
            
            
            
            ##############this is for every comment
            
            if len(pSet)==0:#there is no herf link in the p, normal case
            
                for buyer in buyers:#one buyer is one comment
                    lineList=buyer.xpath('.//text()')
                    for line in lineList:
                        
                        
                        line=line.replace("\n"," ")#two new lines to space
                        line=line.replace("  "," ")#remove new line
                        tagLi=False
                        
                        if not len(line.split())==0:#check whether it is null line
                            if line in liSet:
                                tagLi=True         
                                
                                               
                            divdieAndWrite(line,tagLi,file, tokenizer)
                        
                    file.write("####comment#12345#"+os.linesep)
                    
                    
            else:
                for buyer in buyers:#one buyer is one comment
                    lineList=buyer.xpath('.//text()')
                    
                    lineListlen=len(lineList)
                    beforeTextforP=""
                    
                    index=0
                    #for index in range(lineListlen):
                    while index<lineListlen:
                        
                        
                        ###########set line and lineNext
                        line=lineList[index]
                        #if "Automatically refresh" in line:
                        #   print("bingo")
                        
                        
                        if index+1 <lineListlen:#the is there a lineNext
                            lineNext=lineList[index+1]
                        else:
                            lineNext="abcdefg12345"
                            
                        if "report with a new problem" in line:
                            print("bingo")
                            
                            
                        if lineNext not in pSet:#the normal case
                            line=beforeTextforP + line.replace("\n"," ")#two new lines to space
                            line=line.replace("  "," ")#remove new line
                            
                                
                            
                            tagLi=False
                            if not len(line.split())==0:#check whether it is null line
                                if "Rebuild Project" in line:
                                        print("bingo")
                                if line in liSet:
                                    tagLi=True
                                
                                
                                if not beforeTextforP=="":#initial again
                                    lineRmChk=line
                                    for item in liList:####for the code part of the li
                                        if item in lineRmChk and lineRmChk.strip().startswith(item.strip()):
                                            lineRmChk=lineRmChk.replace(item,'')
                                    if len(lineRmChk.strip())==0:
                                        tagLi=True
                                
                                '''
                                if not beforeTextforP=="":#initial again
                                    i=0
                                    for item in liSet:####for the code part of the li
                                        if item in line:
                                            i=i+1
                                            if i==2:
                                                tagLi=True
                                                break         
                                '''
                                divdieAndWrite(line,tagLi,file, tokenizer)
                                
                            if not beforeTextforP=="":#initial again
                                beforeTextforP=""
                            index=index+1
                        else:
                            beforeTextforP=beforeTextforP+" "+line+" "+lineNext+" "
                            index=index+2
                            
                            
                            
                            
                            
                    file.write("####comment#12345#"+os.linesep)
                    '''
                    for line in lineList:#every p or li in a comment
                        
                        #if "can raise it." in line:
                        #    print("bingo")
                        
                        line=line.replace("\n"," ")#two new lines to space
                        line=line.replace("  "," ")#remove new line
                        tagLi=False
                        
                        if not len(line.split())==0:#check whether it is null line
                            if line in liSet:
                                tagLi=True         
                                
                                               
                            divdieAndWrite(line,tagLi,file, tokenizer)
                        
                    file.write("####comment#12345#"+os.linesep)
                    '''
                
            
            
            file.close()    
            print(suballDir)
            
            
            '''
            titleText=""
                
            for titleLine in titlelines:
                if not len(titleLine.split())==0:#check whether it is null line, title is very easy between two null lines
                    titleText=titleLine
            '''
            
            #aa=bua.xpath('.//p/.//text()')
            #print(aa.text_content())
            
            
            
            
            '''
            #if allDir=="google code":
            if True:
                #tree = html.parse("recdroid-dataset/"+allDir+"/"+suballDir)
                
                tree = html.parse("/home/yu/Downloads/labeledtest_checked/labeledtest_checked/18.html")
                ######every comments
                buyers = tree.xpath('//markdown-widget[@text="comment.content"]')
                ######li part
                liTag=tree.xpath('//markdown-widget[@text="comment.content"]/.//li')
                ######title part
                titleXpath=tree.xpath('//div[@id="gca-project-header"]/p[@class="ng-binding"]')
            
            else:
                
                tree = html.parse("recdroid-dataset/"+allDir+"/"+suballDir)
                #tree=html.parse("/home/yu/Downloads/labeledtest_checked/labeledtest_checked/k9#2019.html")
                ######every comments
                buyers = tree.xpath('//td')
                ######li part
                liTag=tree.xpath('//td/.//li')
                ######title part
                titleXpath=tree.xpath('//span[@class="js-issue-title"]')
                
                
            titlelines=titleXpath[0].text_content().splitlines()
                
            titleText=""
                
            for titleLine in titlelines:
                if not len(titleLine.split())==0:#check whether it is null line, title is very easy between two null lines
                    titleText=titleLine
            
            for liItem in liTag:
                liTextList=liItem.text_content().splitlines()# I do not know. It may has more than one line.
                for liText in liTextList:
                    liSet.add(liText)#here is the set
                        
                    
                
                
                #liList=liTag[0].text_content().splitlines()
                
            
            ##############this is for the title
            file.write("#####title#12345#"+os.linesep)
            
            divdieAndWrite(titleText,False,file, tokenizer)

            file.write("#####title#12345#"+os.linesep)
            
            
            
            
            ##############this is for every comment
            for buyer in buyers:   #one buyer one comment
                lineList=buyer.text_content().splitlines()
                
                for line in lineList:
                    
                    tagLi=False
                    
                    if not len(line.split())==0:#check whether it is null line
                    
                        if line in liSet:
                            tagLi=True
                
                        divdieAndWrite(line,tagLi,file, tokenizer)
                file.write("####comment#12345#"+os.linesep)
            
            
            
                
            file.close()    
            print suballDir
            '''
                    #processLine(sentence,x, y, z)


tokenizer = PunktSentenceTokenizer()
pathDir=os.listdir("recdroid-dataset")
for allDir in pathDir:
    subpathDir=os.listdir("recdroid-dataset/"+allDir)
    TrackerFolders(subpathDir,tokenizer)

