import spacy
import contextualSpellCheck

# ENGLISH

# python -m spacy download en_core_web_sm
# nlp = spacy.load("en_core_web_sm")

# nlp.add_pipe("contextual spellchecker", config={"max_edit_dist": 100})

# doc = nlp("Icome was $9.4 milion compared to the last yiear of $2.7 milion.")
# print(doc._.performed_spellCheck)
# print(doc._.outcome_spellCheck)

# # Doc Extention
# print(doc._.contextual_spellCheck)

# print(doc._.performed_spellCheck)

# print(doc._.suggestions_spellCheck)

# print(doc._.outcome_spellCheck)

# print(doc._.score_spellCheck)

# # Token Extention
# print(doc[4]._.get_require_spellCheck)

# print(doc[4]._.get_suggestion_spellCheck)

# print(doc[4]._.score_spellCheck)

# # Span Extention
# print(doc[2:6]._.get_has_spellCheck)

# print(doc[2:6]._.score_spellCheck)

# JAPANESE

# python -m spacy download ja_core_news_sm
# pip install mecab-python3==0.996.5
# pip install ipadic==1.0.0
# pip install unidic-lite==1.0.6
# pip install fugashi==1.1.0
# nlp = spacy.load("ja_core_news_sm")

# nlp.add_pipe(
#     "contextual spellchecker",
#     config={
#         "model_name": "cl-tohoku/bert-base-japanese-whole-word-masking",
#         "max_edit_dist": 2,
#     },
# )

# doc = nlp("しかし大勢においては、ここような事故はウィキペディアの拡大には影響を及ぼしていない。")
# print(doc._.performed_spellCheck)
# print(doc._.outcome_spellCheck)

# PORTUGUESE

# python -m spacy download pt_core_news_lg
# nlp = spacy.load("pt_core_news_lg")

# nlp.add_pipe(
#     "contextual spellchecker",
#     config={
#         "model_name": "neuralmind/bert-large-portuguese-cased",
#         "max_edit_dist": 5,
#     },
# )

## PEDIR DINHEIRO EMPRESTADO ÀS PESSOAS DA ALDEIA
# doc = nlp("EDIR DINHEIRO EMPRESTADO ÀS PESSOAS DO ALDEIRA".lower())
# print(doc._.performed_spellCheck)
# print(doc._.outcome_spellCheck)


# SPANISH

# # python -m spacy download es_dep_news_trf
# nlp = spacy.load("es_dep_news_trf")

# nlp.add_pipe(
#     "contextual spellchecker",
#     config={
#         "model_name": "Geotrend/bert-base-es-cased",
#         "max_edit_dist": 5,
#     },
# )

## HABITAN EN AGUAS POCO PROFUNDAS Y ROCOSAS
# doc = nlp("HABITAN AGUAS POCO PROFUNDAS Y ROCOSAS".lower())
# print(doc._.performed_spellCheck)
# print(doc._.outcome_spellCheck)


# FRENCH

# python -m spacy download fr_core_news_sm
nlp = spacy.load("fr_core_news_sm")

nlp.add_pipe(
    "contextual spellchecker",
    config={
        "model_name": "camembert-base",
        "max_edit_dist": 5,
    },
)

# CE SITE CONTIENT QUATRE TOMBEAUX DE LA DYNASTIE ACHÉMÉNIDE ET SEPT DES SASSANIDES.
doc = nlp("CE SITE CONTIENT QUATRE TOMBEAUX DE LA DYNASTIE ASHÉMÉNIDE ET SEPT DES SASANNIDES".lower())
doc = nlp("CE SITE CONTIENT QUATRE TOMBEAUX".lower())
print(doc._.performed_spellCheck)
print(doc._.outcome_spellCheck)
