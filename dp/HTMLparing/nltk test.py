from nltk.tokenize import sent_tokenize, word_tokenize
 
data = "App crash: RuntimeException: MessageId is null. You must call setMessageUuid(...)"
print(sent_tokenize(data))




data = "1. During multiple input of identical values e.g. 10 km (by day) and 10 litres some calculations fail. E.g. first calculates correcty 100 l/km but then next two results are \"infinity l / 100 km. Plotting will death lock program then and you have to restart device."
print(sent_tokenize(data))
