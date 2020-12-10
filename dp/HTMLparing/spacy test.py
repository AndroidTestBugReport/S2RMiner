import spacy
from spacy.lang.en import English

'''
nlp = English()  # just the language with no model
sentencizer = nlp.create_pipe("sentencizer")
nlp.add_pipe(sentencizer)
'''
nlp = spacy.load("en_core_web_sm")


#nlp = spacy.load("en_core_web_sm")
doc = nlp(u"App crash: runtime exception: message id is null.")
for sentence in doc.sents:
    print(sentence.text)
    print("lunar")
    
doc = nlp(u"I really don't know what could cause this because I haven't touched anything related to GPS at all. Some have reported that the app \"GPS Test\" fix the issue when they let it run for a few minutes.")
for sentence in doc.sents:
    print(sentence.text)
    print("lunar")
    
    
doc = nlp(u"App crash: runtime exception: message id is null. You must call set messageUuid(...)")
for sentence in doc.sents:
    print(sentence.text)
    print("lunar")
    
doc = nlp(u"I want to use an internal email adress like \"foo@b.2\". We use some custom software on our own servers only for internal communication. It works in Thunderbird.")
for sentence in doc.sents:
    print(sentence.text)
    print("lunar")
    

doc = nlp(u"    (All messages and texts are in german on my device; I don't know the texts if you use K9 in english. I just tried to translate them)")
for sentence in doc.sents:
    print(sentence.text)
    print("lunar")
    
    
doc = nlp(u"By the way (I seen it before): It is strange to meet button 'Load default' on the page 'DB default settings'.")
for sentence in doc.sents:
    print(sentence.text)
    print("lunar")
    
doc = nlp(u"Finally found out the issue. Modified.")
for sentence in doc.sents:
    print(sentence.text)
    print("lunar")
    
doc = nlp(u"I confirm. No more crashes.")
for sentence in doc.sents:
    print(sentence.text)
    print("lunar")
    
doc = nlp(u"DB default settings - Crash")
for sentence in doc.sents:
    print(sentence.text)
    print("lunar")
    
    
doc = nlp(u"E/AndroidRuntime(  727): java.lang.RuntimeException: An error occured while executing doInBackground()")
for sentence in doc.sents:
    print(sentence.text)
    print("lunar")


doc = nlp(u"Please close this issue. This issue is reproducible only in Play store version. I think the issue has been fixed in the current repository.")
for sentence in doc.sents:
    print(sentence.text)
    print("lunar")

doc = nlp(u"java.lang.NullPointerException: Attempt to invoke virtual method 'android.content.Context android.view.View.getContext()' on a null object reference")
for sentence in doc.sents:
    print(sentence.text)
    print("lunar")

doc = nlp(u"I don't see value in K-9 supporting the defederation of email like this. I also don't see Thunderbird on its own as a good enough reason to ignore that. I suspect Thunderbird is lack of validation not choice to support this. So it's not a feature to have parity with, more exploitation  of a design flaw.")
for sentence in doc.sents:
    print(sentence.text)
    print("lunar")
    
doc = nlp(u"1. During multiple input of identical values e.g. 10 km (by day) and 10 litres some calculations fail. E.g. first calculates correcty 100 l/km but then next two results are \"infinity l / 100 km. Plotting will death lock program then and you have to restart device.")
for sentence in doc.sents:
    print(sentence.text)
    print("lunar")
