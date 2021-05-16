from autocorrect import Speller

spell = Speller('pl')
print(spell('ptaaki latatją kluczmm'))

spell = Speller('fr')
print(spell("CE SITE CONTIENT QUATRE TOMBEAUX DE LA DYNASTIE ASHÉMÉNIDE ET SEPT DES SASANNIDES".lower()))