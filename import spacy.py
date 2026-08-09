import spacy
import spacy.cli
import pandas as pd
import stanza
import os

#------------------------------------------------------------------------------------------------------------
spacy.cli.download("fr_core_news_lg")
stanza.download("fr")
nlp1 = stanza.Pipeline("fr",processors="tokenize,pos,lemma,depparse")

nlp2 = spacy.load("fr_core_news_lg")
#------------------------------------------------------------------------------------------------------------
text = "J’ai l’impression que pour moi le problème est plus l’énergie mentale car souvent je me sens trop épuisé pour faire des choses qui demandent de l’effort mental :see_no_evil: Puis je suis pas assez passionné par les langues pour les traiter de responsabilité, j’apprends une langue jusqu’à ce qu’elle m’est amusante Sinon d’habitude j’apprenais les langues en bavardant en ligne et en regardant du contenu genre sur YT ou Insta, mais ironiquement c’est précisément ça ce que j’essaie d’éviter ces jours-ci, du coup je galère un peu à trouver des moyens d’utiliser la langue que j’apprends sans sentir que je perds du temps"
doc_stan = nlp1(text)
spacy_s = nlp2(text)
data = []
stanza_w = []
for sentence in doc_stan.sentences:
    for word in sentence.words:
        stanza_w.append(word)

#--------------------------------------------------------------------------------------------------------------
for spacy_token, stanza_sent in zip (spacy_s, stanza_w):
    data.append({"Tkn": spacy_token.text,"Stlem": stanza_sent.lemma,"SpPOS": spacy_token.pos_,"StPOS": stanza_sent.upos, "Stlem": stanza_sent.lemma,"SpMor": str(spacy_token.morph).replace("|", ",\n"),"StMor": stanza_sent.feats.replace("|", ",\n") if stanza_sent.feats else "","SpDep": spacy_token.dep_,"StDep": stanza_sent.deprel,"SpH": spacy_token.head.text})
df= pd.DataFrame(data)
df_main = df.drop(columns=["SpMor", "StMor"])
df_morph = df[["Tkn", "SpMor", "StMor"]]
with open("bot_test.txt", "w") as f:
    f.write(df.to_markdown(index=False))
    #f.write(df_main.to_markdown(index=False))
    #f.write("\n\n")
    #f.write(df_morph.to_markdown(index=False))
os.startfile("bot_test.txt")
#---------------------------------------------------------------------------------------------------------------