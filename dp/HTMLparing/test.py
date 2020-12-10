from lxml import html
from lxml import etree


tree = html.parse("labeled/google code/wholeId-28-page-1-appid-5.html")
#print(html.tostring(tree))
buyers = tree.xpath('//markdown-widget[@text="comment.content"]')
print buyers[0].text_content().splitlines()[1]
print etree.tostring(buyers[0], pretty_print=True)

def replaceNumDot(x):
    index=x.find(". ")
    if x[index-1].isdigit():
        x=x.replace(x[index-1]+"." , "numDot")

    return x


def processLine(x,senList, stepList, oracleList):
    index=x.find(". ")
    if index==-1: #there is no any ". "left in the sentence
        if x!="":
            
            if x.find("#step")!=-1:# x.contains #step# such as a b c d e #step#
                stepList.append("1")
                x=x.replace("#step"," ")
            else:
                stepList.append("0")

        
        
            
            if x.find("#oracle")!=-1:# x.contains #oracle#
                oracleList.append("1")
                x=x.replace("#oracle"," ")
            else:
                oracleList.append("0")
            #x=x.strip()
            senList.append(x.strip())
        return
    
    
    
    if x[0: index+1]!="" and not x[index-1].isdigit() and not x[index-1]==".":
        add=x[0: index+1];
        add.replace("#step","")
        add.replace("#oracle","")
        senList.append(x[0: index+1].strip())
        
        x=x[x.find(". ")+1:]#cut the x
        
        
        stepIndex=x.find("#step")
        if stepIndex>=0 and stepIndex < 2:#check step
            x=x[6:]
            stepList.append("1")
        else:
            stepList.append("0")
            
        oracleIndex=x.find("#oracle")
        if oracleIndex>=0 and oracleIndex < 2:#check oracle
            x=x[8:]
            oracleList.append("1")
        else:
            oracleList.append("0")
            
    elif x[index-1].isdigit():
        x=x.replace(x[index-1]+"." , "numDot")
    elif x[index-1]==".":
        x=x.replace(x[index-1]+". " , "soOnDot ")

        
        
    processLine(x,senList, stepList, oracleList)
        
    
    
    
    a=[]

#tree = html.parse("labeled/"+"github"+"/"+"appId-3-issueId-3528.html")
#print(html.tostring(tree))
#buyers = tree.xpath('//td')
#print buyers[0].text_content().splitlines()[11]
#print buyers[0].text_content().splitlines()[1]



#x=buyers[0].text_content().splitlines()[11]

x="    Click icon #step"

x=x.strip()

print(x)


senList=[]
stepList=[]
labelList=[]

processLine(x,senList, stepList, labelList)
print("senList")
print(senList)

print("stepList")
print(stepList)

print("labelList")
print(labelList)


print("aa")
