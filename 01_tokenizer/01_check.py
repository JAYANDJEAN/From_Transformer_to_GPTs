import spacy

nlp_en = spacy.load("en_core_web_sm")
nlp_de = spacy.load("de_core_news_sm")
doc = nlp_en("A man in a blue shirt is standing on a ladder cleaning a window.")
print([token.text for token in doc])
doc = nlp_de("Ein Mann in grün hält eine Gitarre, während der andere Mann sein Hemd ansieht.")
print([token.text for token in doc])