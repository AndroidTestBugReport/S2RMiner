import os
import sys
from xml.dom.minidom import Document
from xml.dom import minidom
from lxml import html
from lxml import etree
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
import html2text
import spacy
import re
from spacy.lang.en import English

'''
#nltk
def divdieAndWrite(line,tagLi,file, tokenizer):
    
    
    try:
        tokenizer.train(line)
    except:
        print("error segment")
        
    for sentence in tokenizer.tokenize(line):
        
        if tagLi:
        
            try:
                file.write("numDot "+sentence.strip()+os.linesep)
            except:    
                print("error symbol")
            #print sentence.strip()
            
            
        else:
            
            try:
                file.write(sentence.strip()+os.linesep)
            except:    
                print("error symbol")
            #print sentence.strip()
'''
#spacy
def divdieAndWrite(line,tagLi,file, tokenizer):
    
    
    if "Loosing 3g ramdomly" in line:
        print("bingo")
    
    doc = nlp(str(line))###bin pyhton2 it is the unicode
    
    
    if line.strip()=="|":
        return
    
    
    
    try:
        for sentence in doc.sents:
            
            
            
            if tagLi:
            
                try:
                    if not len(sentence.text.split())==0:
                    
                    
                        #file.write(" #numDot "+sentence.text.strip().encode("utf-8")+os.linesep)
                        file.write(" numDot "+str(sentence.text.strip())+os.linesep)
                        
                except:    
                    print("error symbol")
                #print sentence.strip()
                                
            else:
                
                try:
                    if not len(sentence.text.split())==0:
                        file.write(str(sentence.text.strip())+os.linesep)
                except:    
                    print("error symbol")
    except:
        try:
            file.write(str(line.strip())+os.linesep)#some special symbol
        except:
            print("error symbol")
            
def lineRemove(line,reList):
    for reItem in reList:
        if line.strip().find(reItem)>2:
            line=line.replace(reItem, '\n numDot ' )
        
        line=line.replace(reItem,' numDot ')
    return line

def TrackerFolders(subpathDir, tokenizer, nlp):
        h = html2text.HTML2Text()
        h.ignore_links=True
        h.body_width = 0
        h.ignore_emphasis=True
        h.ignore_anchors = True
        #h.skip_internal_links = True
        #h.inline_links = True
        h.ignore_images = True
    
        for suballDir in subpathDir:
            print(suballDir)

            file = open("recdroid-txt/"+allDir+"/"+suballDir[:-5]+".txt", 'w')
            
            
            liSet=set([])
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
            
            
            if allDir=="github":
                tree = html.parse("recdroid-dataset/"+allDir+"/"+suballDir)
                ######every comments                
                buyers=tree.xpath('//td')
                ######li part
                liTag=tree.xpath('//td/.//li')
                ######title part
                titleXpath=tree.xpath('//span[@class="js-issue-title"]')[0]
                
                
            elif allDir=="google code":
                tree = html.parse("recdroid-dataset/"+allDir+"/"+suballDir)
                
                #tree = html.parse("/home/yu/Downloads/labeledtest_checked/labeledtest_checked/18.html")
                ######every comments
                buyers=tree.xpath('//markdown-widget[@text="comment.content"]')
                ######li part
                liTag=tree.xpath('//markdown-widget[@text="comment.content"]/.//li')
                ######title part
                titleXpath=tree.xpath('//div[@id="gca-project-header"]/p[@class="ng-binding"]')[0]
            
            else:
                continue
                
            #print(type(etree.tostring(titleXpath)))
            ##################take title text
            titleText=h.handle(etree.tostring(titleXpath).decode())
            
            ##############this is for the title
            file.write("#####title#12345#"+os.linesep)
            
            titleList=titleText.splitlines()
            for eachTitleLine in titleList:
                if not len(eachTitleLine.split())==0:
            
                    #divdieAndWrite(eachTitleLine,False,file, tokenizer)
                    divdieAndWrite(eachTitleLine,False,file, nlp)
            file.write("#####title#12345#"+os.linesep)
            
            
            
            ##################take li text
            for liItem in liTag:
                
                liTextlines=h.handle(etree.tostring(liItem).decode())
                liTextList=liTextlines.splitlines()
                for liText in liTextList:
                    if not len(liText.split())==0:
                        liSet.add(liText.strip())#here is the set
                        
                        
                        
                        
            ##############this is for every comment
            for buyer in buyers:#one buyer is one comment
                lineList=h.handle(etree.tostring(buyer).decode()).splitlines()
                for line in lineList:
                    tagLi=False
                    if not len(line.split())==0:#check whether it is null line
                        
                        if "Loosing 3g ramdomly" in line:
                            print("bingo")
                        
                        if line.strip() in liSet:    #we do not use this in this version. we use a heuristic
                            tagLi=True
                            line=line.strip()
                            
                            
                        numList0=re.findall(r' +\d+\\\. ',line)#this is for 1\.
                        numList1=re.findall(r' +\d+\. ',line)#this is for 1.
                        
                        if not len(numList0)==0:
                            line=lineRemove(line, numList0)
                            
                            
                        if not len(numList1)==0:
                            line=lineRemove(line, numList1)
                        
                        
                        
                        
                        '''below is a old version
                        if index>0 and line[index-1].isdigit() and index<6:#6 is for hundreds . This part is for <li>
                            line=line.replace(line[:index]+"." , "numDot") 
                        '''
                        
                         
                        #divdieAndWrite(line,tagLi,file, tokenizer)
                        divdieAndWrite(line,tagLi,file, nlp)
                file.write("####comment#12345#"+os.linesep)
            
            file.close()    
            print(suballDir)
            
            
            
                    #processLine(sentence,x, y, z)


tokenizer = PunktSentenceTokenizer()
#nlp = spacy.load("en_core_web_sm")
nlp = English()  # just the language with no model
sentencizer = nlp.create_pipe("sentencizer")
nlp.add_pipe(sentencizer)

pathDir=os.listdir("recdroid-dataset")
for allDir in pathDir:
    subpathDir=os.listdir("recdroid-dataset/"+allDir)
    TrackerFolders(subpathDir,tokenizer,nlp)

