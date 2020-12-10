from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
text = "There are more options under Mobile Networks in 2.3.4 that should do what you're asking, so you will see it in RC2. Thanks for the submission!"
tokenizer = PunktSentenceTokenizer()
tokenizer.train(text)
print(tokenizer.tokenize(text))


