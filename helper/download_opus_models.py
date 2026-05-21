"""
Download-Script für gefilterte OPUS-MT Modelle
Lädt alle Modelle in den Ordner: ./opus_mt_models/<model-name>/

Nutzung:
    python download_opus_models.py
    python download_opus_models.py --output ./mein_ordner
    python download_opus_models.py --only-big        # nur tc-big Modelle
    python download_opus_models.py --dry-run         # nur anzeigen, nicht laden

Abhängigkeiten:
    pip install huggingface_hub
"""

import argparse
import os
import sys
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# Gefilterte Modell-Liste
# Kriterien: Beide Seiten (src + tgt) gehören zu den 16 Zielsprachen:
# en, de, fr, zh, ru, uk, es, pt, tr, it, ja, sv, pl, ko, nl, vi
# (inkl. Gruppenmodelle: zle, itc, gmq, gmw, zlw, roa, ROMANCE)
# ──────────────────────────────────────────────────────────────────────────────

#opus_mt_models/Helsinki-NLP__opus-mt-tc-big-gmw-gmw
langs_to_model = {
        ("en","de"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-gmw-gmw",
        ("en","fr"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-en-fr",
        ("en","zh"): "opus_mt_models/Helsinki-NLP__opus-mt-en-zh",
        ("en","ru"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-en-zle", #mt-en-ru
        ("en","uk"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-en-zle", #mt-en-uk
        ("en","es"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-en-es",
        ("en","pt"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-en-pt",
        ("en","tr"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-en-tr",
        ("en","it"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-en-it",
        ("en","ja"): "opus_mt_models/Helsinki-NLP__opus-tatoeba-en-ja", # opus_mt_models/opus-tatoeba-en-ja 
        ("en","sv"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-en-gmq", #opus-mt-en-sv
        ("en","pl"): "opus_mt_models/Helsinki-NLP__opus-mt-en-zlw", # / opus-mt-tc-big-zlw-en / opus-mt-en-pl
        ("en","ko"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-en-ko",
        ("en","nl"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-gmw-gmw", #opus-mt-en-nl
        ("en","vi"): "opus_mt_models/Helsinki-NLP__opus-mt-en-vi", 
        
        ("de","en"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-gmw-gmw",
        ("de","fr"): "opus_mt_models/Helsinki-NLP__opus-mt-de-fr",
        ("de","zh"): "",
        ("de","ru"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-de-zle",
        ("de","uk"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-de-zle",
        ("de","es"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-de-es",
        ("de","pt"): "",
        ("de","tr"): "",
        ("de","it"): "opus_mt_models/Helsinki-NLP__opus-mt-de-it",
        ("de","ja"): "",  #back track?
        ("de","sv"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-de-gmq",
        ("de","pl"): "opus_mt_models/Helsinki-NLP__opus-mt-de-pl",
        ("de","ko"): "",
        ("de","nl"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-gmw-gmw",
        ("de","vi"): "opus_mt_models/Helsinki-NLP__opus-mt-de-vi",
        
        ("fr","en"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-fr-en",
        ("fr","de"): "opus_mt_models/Helsinki-NLP__opus-mt-fr-de",
        ("fr","zh"): "",
        ("fr","ru"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-fr-zle", #test
        ("fr","uk"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-fr-zle",
        ("fr","es"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-itc-itc",
        ("fr","pt"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-itc-itc",
        ("fr","tr"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-itc-tr",
        ("fr","it"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-itc-itc",
        ("fr","ja"): "",  #back track?
        ("fr","sv"): "opus_mt_models/Helsinki-NLP__opus-mt-fr-sv", ################################################################
        ("fr","pl"): "opus_mt_models/Helsinki-NLP__opus-mt-fr-pl",
        ("fr","ko"): "",
        ("fr","nl"): "",
        ("fr","vi"): "opus_mt_models/Helsinki-NLP__opus-mt-fr-vi",
        
        ("ru","en"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-zle-en",
        ("ru","de"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-zle-de",
        ("ru","fr"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-zle-fr",
        ("ru","zh"): "",
        ("ru","uk"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-zle-zle",
        ("ru","es"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-zle-es",
        ("ru","pt"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-zle-pt",
        ("ru","tr"): "",
        ("ru","it"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-zle-it",
        ("ru","ja"): "",  #back track?
        ("ru","sv"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-zle-gmq",
        ("ru","pl"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-zle-zlw",
        ("ru","ko"): "",
        ("ru","nl"): "",
        ("ru","vi"): "opus_mt_models/Helsinki-NLP__opus-mt-ru-vi", 
        
        ("uk","en"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-zle-en",
        ("uk","de"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-zle-de",
        ("uk","fr"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-zle-fr",
        ("uk","zh"): "",
        ("uk","ru"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-zle-zle",
        ("uk","es"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-zle-es",
        ("uk","pt"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-zle-pt",
        ("uk","tr"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-base-uk-tr",
        ("uk","it"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-zle-it",
        ("uk","ja"): "",  #back track?
        ("uk","sv"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-zle-gmq",
        ("uk","pl"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-zle-zlw",
        ("uk","ko"): "",
        ("uk","nl"): "opus_mt_models/Helsinki-NLP__opus-mt-uk-nl",
        ("uk","vi"): "", 
        
        ("es","en"): "opus_mt_models/Helsinki-NLP__opus-mt-es-en",
        ("es","de"): "opus_mt_models/Helsinki-NLP__opus-mt-es-de",
        ("es","fr"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-itc-itc",
        ("es","zh"): "",
        ("es","ru"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-es-zle",
        ("es","uk"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-es-zle",
        ("es","pt"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-itc-itc",
        ("es","tr"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-itc-tr",
        ("es","it"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-itc-itc",
        ("es","ja"): "", #back track?
        ("es","sv"): "",
        ("es","pl"): "opus_mt_models/Helsinki-NLP__opus-mt-es-pl",
        ("es","ko"): "",
        ("es","nl"): "opus_mt_models/Helsinki-NLP__opus-mt-es-nl",
        ("es","vi"): "opus_mt_models/Helsinki-NLP__opus-mt-es-vi", 
        
        ("nl","en"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-gmw-gmw",
        ("nl","de"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-gmw-gmw",
        ("nl","fr"): "opus_mt_models/Helsinki-NLP__opus-mt-nl-fr",
        ("nl","es"): "opus_mt_models/Helsinki-NLP__opus-mt-nl-es",
        ("nl","uk"): "opus_mt_models/Helsinki-NLP__opus-mt-nl-uk",
        ("nl","sv"): "opus_mt_models/Helsinki-NLP__opus-mt-nl-sv",
        ("nl","it"): "",
        ("nl","pt"): "",
        ("nl","ru"): "",
        ("nl","pl"): "",
        ("nl","tr"): "",
        ("nl","zh"): "",
        ("nl","ja"): "", #back track wie?
        ("nl","ko"): "",
        ("nl","vi"): "",

        ("pt","en"): "opus_mt_models/Helsinki-NLP__opus-mt-itc-en",
        ("pt","fr"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-itc-itc",
        ("pt","es"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-itc-itc",
        ("pt","it"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-itc-itc",
        ("pt","ru"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-pt-zle",
        ("pt","uk"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-pt-zle",
        ("pt","de"): "",
        ("pt","sv"): "",
        ("pt","pl"): "",
        ("pt","nl"): "",
        ("pt","tr"): "",
        ("pt","zh"): "",
        ("pt","ja"): "", #back track wie?
        ("pt","ko"): "",
        ("pt","vi"): "",

        ("tr","en"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-tr-en",
        ("tr","fr"): "opus_mt_models/Helsinki-NLP__opus-mt-tr-fr",
        ("tr","es"): "opus_mt_models/Helsinki-NLP__opus-mt-tr-es",
        ("tr","uk"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-base-tr-uk",
        ("tr","sv"): "opus_mt_models/Helsinki-NLP__opus-mt-tr-sv",
        ("tr","de"): "",
        ("tr","it"): "",
        ("tr","pt"): "",
        ("tr","ru"): "",
        ("tr","pl"): "",
        ("tr","nl"): "",
        ("tr","zh"): "",
        ("tr","ja"): "", #back track wie?
        ("tr","ko"): "",
        ("tr","vi"): "",

        ("it","en"): "opus_mt_models/Helsinki-NLP__opus-mt-itc-en",
        ("it","de"): "opus_mt_models/Helsinki-NLP__opus-mt-it-de",
        ("it","fr"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-itc-itc",
        ("it","es"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-itc-itc",
        ("it","pt"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-itc-itc",
        ("it","ru"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-it-zle",
        ("it","uk"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-it-zle",
        ("it","sv"): "opus_mt_models/Helsinki-NLP__opus-mt-it-sv",
        ("it","tr"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-itc-tr",
        ("it","vi"): "opus_mt_models/Helsinki-NLP__opus-mt-it-vi",
        ("it","ja"): "", #back track wie?
        ("it","pl"): "",
        ("it","nl"): "",
        ("it","zh"): "",
        ("it","ko"): "",

        ("sv","en"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-gmq-en",
        ("sv","fr"): "opus_mt_models/Helsinki-NLP__opus-mt-sv-fr",
        ("sv","es"): "opus_mt_models/Helsinki-NLP__opus-mt-sv-es",
        ("sv","ru"): "opus_mt_models/Helsinki-NLP__opus-mt-sv-ru",
        ("sv","uk"): "opus_mt_models/Helsinki-NLP__opus-mt-sv-uk",
        ("sv","nl"): "opus_mt_models/Helsinki-NLP__opus-mt-sv-nl",
        ("sv","tr"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-gmq-tr",
        ("sv","zh"): "",
        ("sv","de"): "",
        ("sv","it"): "",
        ("sv","pt"): "",
        ("sv","pl"): "",
        ("sv","ja"): "",
        ("sv","ko"): "",
        ("sv","vi"): "",

        ("pl","en"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-zlw-en",
        ("pl","de"): "opus_mt_models/Helsinki-NLP__opus-mt-pl-de",
        ("pl","fr"): "opus_mt_models/Helsinki-NLP__opus-mt-pl-fr",
        ("pl","es"): "opus_mt_models/Helsinki-NLP__opus-mt-pl-es",
        ("pl","ru"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-zlw-zle",
        ("pl","uk"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-zlw-zle",
        ("pl","sv"): "opus_mt_models/Helsinki-NLP__opus-mt-pl-sv",
        ("pl","ja"): "", #back track wie?
        ("pl","it"): "",
        ("pl","pt"): "",
        ("pl","nl"): "",
        ("pl","tr"): "",
        ("pl","zh"): "",
        ("pl","ko"): "",
        ("pl","vi"): "",
    }

langs_to_model = set(langs_to_model.values())
langs_to_model.remove("")

# Modelle die von TalkBridge genutzt werden
# USED_MODELS = []
# for model in langs_to_model:
#     model_name = "".join(model.split("/")[1:])
#     modelprefix = model_name.split("__")[0]
#     model_suffix = model_name.split("__")[1]
#     USED_MODELS.append(modelprefix + "/" + model_suffix)

# print(USED_MODELS)

#Helsinki-NLP/opus-mt-tc-big-zh-ja
ALL_MODELS = [
    # ── tc-big (beste Qualität) ───────────────────────────────────────────────
    "Helsinki-NLP/opus-mt-tc-big-zh-ja",
    "Helsinki-NLP/opus-mt-tc-big-en-it",
    "Helsinki-NLP/opus-mt-tc-big-en-ko",
    "Helsinki-NLP/opus-mt-tc-big-en-es",
    "Helsinki-NLP/opus-mt-tc-big-en-fr",
    "Helsinki-NLP/opus-mt-tc-big-en-pt",
    "Helsinki-NLP/opus-mt-tc-big-en-tr",
    "Helsinki-NLP/opus-mt-tc-big-en-gmq",    # en -> sv/da/no
    "Helsinki-NLP/opus-mt-tc-big-en-zle",    # en -> ru/uk
    # "Helsinki-NLP/opus-mt-tc-big-en-zlw",    # en -> pl
    "Helsinki-NLP/opus-mt-tc-big-en-itc",    # en -> fr/es/it/pt
    "Helsinki-NLP/opus-mt-tc-big-en-gmw",    # en -> de/nl
    "Helsinki-NLP/opus-mt-tc-big-fr-en",
    "Helsinki-NLP/opus-mt-tc-big-it-en",
    "Helsinki-NLP/opus-mt-tc-big-tr-en",
    "Helsinki-NLP/opus-mt-tc-big-ko-en",
    "Helsinki-NLP/opus-mt-tc-big-gmq-en",    # sv/da/no -> en
    "Helsinki-NLP/opus-mt-tc-big-zle-en",    # ru/uk -> en
    "Helsinki-NLP/opus-mt-tc-big-zlw-en",    # pl -> en
    "Helsinki-NLP/opus-mt-tc-big-itc-tr",    # fr/es/it/pt -> tr
    "Helsinki-NLP/opus-mt-tc-big-itc-itc",   # fr/es/it/pt untereinander
    "Helsinki-NLP/opus-mt-tc-big-gmq-itc",   # sv -> fr/es/it/pt
    "Helsinki-NLP/opus-mt-tc-big-gmq-tr",    # sv -> tr
    "Helsinki-NLP/opus-mt-tc-big-gmq-zle",   # sv -> ru/uk
    "Helsinki-NLP/opus-mt-tc-big-gmq-zlw",   # sv -> pl
    "Helsinki-NLP/opus-mt-tc-big-gmq-gmq",   # sv/da/no untereinander
    "Helsinki-NLP/opus-mt-tc-big-zle-it",    # ru/uk -> it
    "Helsinki-NLP/opus-mt-tc-big-zle-pt",    # ru/uk -> pt
    "Helsinki-NLP/opus-mt-tc-big-zle-es",    # ru/uk -> es
    "Helsinki-NLP/opus-mt-tc-big-zle-fr",    # ru/uk -> fr
    "Helsinki-NLP/opus-mt-tc-big-zle-de",    # ru/uk -> de
    "Helsinki-NLP/opus-mt-tc-big-zle-itc",   # ru/uk -> fr/es/it/pt
    "Helsinki-NLP/opus-mt-tc-big-zle-gmq",   # ru/uk -> sv
    "Helsinki-NLP/opus-mt-tc-big-zle-zlw",   # ru/uk -> pl
    "Helsinki-NLP/opus-mt-tc-big-zle-zle",   # ru <-> uk
    "Helsinki-NLP/opus-mt-tc-big-it-zle",    # it -> ru/uk
    "Helsinki-NLP/opus-mt-tc-big-fr-zle",    # fr -> ru/uk
    "Helsinki-NLP/opus-mt-tc-big-pt-zle",    # pt -> ru/uk
    "Helsinki-NLP/opus-mt-tc-big-es-zle",    # es -> ru/uk
    "Helsinki-NLP/opus-mt-tc-big-de-zle",    # de -> ru/uk
    "Helsinki-NLP/opus-mt-tc-big-de-gmq",    # de -> sv
    "Helsinki-NLP/opus-mt-tc-big-de-es",     # de -> es
    "Helsinki-NLP/opus-mt-tc-big-zlw-zle",   # pl -> ru/uk
    "Helsinki-NLP/opus-mt-tc-big-gmw-gmw",   # en/de/nl untereinander

    # ── tc-base (schneller, weniger RAM) ─────────────────────────────────────
    "Helsinki-NLP/opus-mt-tc-base-uk-tr",
    "Helsinki-NLP/opus-mt-tc-base-tr-uk",
    "Helsinki-NLP/opus-mt-tc-base-gmw-gmw",  # en/de/nl untereinander

    # ── Ältere opus-mt Generation ─────────────────────────────────────────────
    # Chinesisch
    "Helsinki-NLP/opus-mt-zh-en",
    "Helsinki-NLP/opus-mt-zh-de",
    "Helsinki-NLP/opus-mt-zh-fr",
    "Helsinki-NLP/opus-mt-zh-it",
    "Helsinki-NLP/opus-mt-zh-nl",
    "Helsinki-NLP/opus-mt-zh-sv",
    "Helsinki-NLP/opus-mt-zh-uk",
    "Helsinki-NLP/opus-mt-zh-vi",
    # Japanisch
    "Helsinki-NLP/opus-mt-ja-en",
    "Helsinki-NLP/opus-mt-ja-de",
    "Helsinki-NLP/opus-mt-ja-fr",
    "Helsinki-NLP/opus-mt-ja-es",
    "Helsinki-NLP/opus-mt-ja-it",
    "Helsinki-NLP/opus-mt-ja-pt",
    "Helsinki-NLP/opus-mt-ja-ru",
    "Helsinki-NLP/opus-mt-ja-pl",
    "Helsinki-NLP/opus-mt-ja-nl",
    "Helsinki-NLP/opus-mt-ja-sv",
    "Helsinki-NLP/opus-mt-ja-tr",
    "Helsinki-NLP/opus-mt-ja-vi",
    # Koreanisch
    "Helsinki-NLP/opus-mt-ko-en",
    "Helsinki-NLP/opus-mt-ko-de",
    "Helsinki-NLP/opus-mt-ko-fr",
    "Helsinki-NLP/opus-mt-ko-es",
    "Helsinki-NLP/opus-mt-ko-ru",
    "Helsinki-NLP/opus-mt-ko-sv",
    # Türkisch
    "Helsinki-NLP/opus-mt-tr-en",
    "Helsinki-NLP/opus-mt-tr-fr",
    "Helsinki-NLP/opus-mt-tr-es",
    "Helsinki-NLP/opus-mt-tr-sv",
    "Helsinki-NLP/opus-mt-tr-uk",
    # Ukrainisch
    "Helsinki-NLP/opus-mt-uk-en",
    "Helsinki-NLP/opus-mt-uk-de",
    "Helsinki-NLP/opus-mt-uk-fr",
    "Helsinki-NLP/opus-mt-uk-es",
    "Helsinki-NLP/opus-mt-uk-it",
    "Helsinki-NLP/opus-mt-uk-pt",
    "Helsinki-NLP/opus-mt-uk-pl",
    "Helsinki-NLP/opus-mt-uk-ru",
    "Helsinki-NLP/opus-mt-uk-sv",
    "Helsinki-NLP/opus-mt-uk-nl",
    "Helsinki-NLP/opus-mt-uk-tr",
    # Russisch
    "Helsinki-NLP/opus-mt-ru-en",
    "Helsinki-NLP/opus-mt-ru-fr",
    "Helsinki-NLP/opus-mt-ru-es",
    "Helsinki-NLP/opus-mt-ru-sv",
    "Helsinki-NLP/opus-mt-ru-uk",
    "Helsinki-NLP/opus-mt-ru-vi",
    # Vietnamesisch
    "Helsinki-NLP/opus-mt-vi-en",
    "Helsinki-NLP/opus-mt-vi-de",
    "Helsinki-NLP/opus-mt-vi-fr",
    "Helsinki-NLP/opus-mt-vi-es",
    "Helsinki-NLP/opus-mt-vi-it",
    "Helsinki-NLP/opus-mt-vi-ru",
    # Polnisch
    "Helsinki-NLP/opus-mt-pl-en",
    "Helsinki-NLP/opus-mt-pl-de",
    "Helsinki-NLP/opus-mt-pl-fr",
    "Helsinki-NLP/opus-mt-pl-es",
    "Helsinki-NLP/opus-mt-pl-sv",
    "Helsinki-NLP/opus-mt-pl-uk",
    # Niederländisch
    "Helsinki-NLP/opus-mt-nl-en",
    "Helsinki-NLP/opus-mt-nl-fr",
    "Helsinki-NLP/opus-mt-nl-es",
    "Helsinki-NLP/opus-mt-nl-sv",
    "Helsinki-NLP/opus-mt-nl-uk",
    # Schwedisch
    "Helsinki-NLP/opus-mt-sv-en",
    "Helsinki-NLP/opus-mt-sv-fr",
    "Helsinki-NLP/opus-mt-sv-es",
    "Helsinki-NLP/opus-mt-sv-nl",
    "Helsinki-NLP/opus-mt-sv-ru",
    "Helsinki-NLP/opus-mt-sv-uk",
    # Italienisch
    "Helsinki-NLP/opus-mt-it-en",
    "Helsinki-NLP/opus-mt-it-de",
    "Helsinki-NLP/opus-mt-it-fr",
    "Helsinki-NLP/opus-mt-it-es",
    "Helsinki-NLP/opus-mt-it-sv",
    "Helsinki-NLP/opus-mt-it-uk",
    "Helsinki-NLP/opus-mt-it-vi",
    # Französisch
    "Helsinki-NLP/opus-mt-fr-en",
    "Helsinki-NLP/opus-mt-fr-de",
    "Helsinki-NLP/opus-mt-fr-es",
    "Helsinki-NLP/opus-mt-fr-it",
    "Helsinki-NLP/opus-mt-fr-nl",
    "Helsinki-NLP/opus-mt-fr-pl",
    "Helsinki-NLP/opus-mt-fr-ru",
    "Helsinki-NLP/opus-mt-fr-sv",
    "Helsinki-NLP/opus-mt-fr-uk",
    "Helsinki-NLP/opus-mt-fr-vi",
    # Spanisch
    "Helsinki-NLP/opus-mt-es-en",
    "Helsinki-NLP/opus-mt-es-de",
    "Helsinki-NLP/opus-mt-es-fr",
    "Helsinki-NLP/opus-mt-es-it",
    "Helsinki-NLP/opus-mt-es-nl",
    "Helsinki-NLP/opus-mt-es-pl",
    "Helsinki-NLP/opus-mt-es-ru",
    "Helsinki-NLP/opus-mt-es-uk",
    "Helsinki-NLP/opus-mt-es-vi",
    # Deutsch
    "Helsinki-NLP/opus-mt-de-en",
    "Helsinki-NLP/opus-mt-de-fr",
    "Helsinki-NLP/opus-mt-de-es",
    "Helsinki-NLP/opus-mt-de-it",
    "Helsinki-NLP/opus-mt-de-nl",
    "Helsinki-NLP/opus-mt-de-pl",
    "Helsinki-NLP/opus-mt-de-uk",
    "Helsinki-NLP/opus-mt-de-vi",
    # Englisch (Gruppenmodelle)
    "Helsinki-NLP/opus-mt-en-zh",
    "Helsinki-NLP/opus-mt-en-fr",
    "Helsinki-NLP/opus-mt-en-de",
    "Helsinki-NLP/opus-mt-en-es",
    "Helsinki-NLP/opus-mt-en-it",
    "Helsinki-NLP/opus-mt-en-pt",
    "Helsinki-NLP/opus-mt-en-nl",
    "Helsinki-NLP/opus-mt-en-pl",
    "Helsinki-NLP/opus-mt-en-sv",
    "Helsinki-NLP/opus-mt-en-ru",
    "Helsinki-NLP/opus-mt-en-uk",
    "Helsinki-NLP/opus-mt-en-vi",
    "Helsinki-NLP/opus-mt-en-ko",
    "Helsinki-NLP/opus-mt-en-itc",   # en -> fr/es/it/pt (multilingual)
    "Helsinki-NLP/opus-mt-en-zle",   # en -> ru/uk (multilingual)
    "Helsinki-NLP/opus-mt-en-gmq",   # en -> sv/da/no (multilingual)
    "Helsinki-NLP/opus-mt-en-gmw",   # en -> de/nl (multilingual)
    "Helsinki-NLP/opus-mt-en-zlw",   # en -> pl (multilingual)
    "Helsinki-NLP/opus-mt-en-ROMANCE",
    # Gruppen -> Englisch
    "Helsinki-NLP/opus-mt-roa-en",   # fr/es/pt/it -> en
    "Helsinki-NLP/opus-mt-zle-en",   # ru/uk -> en
    "Helsinki-NLP/opus-mt-zlw-en",   # pl -> en
    "Helsinki-NLP/opus-mt-gmw-en",   # de/nl -> en
    "Helsinki-NLP/opus-mt-gmq-en",   # sv -> en
    "Helsinki-NLP/opus-mt-itc-en",   # fr/es/it/pt -> en
    # Gruppen untereinander
    "Helsinki-NLP/opus-mt-zle-zle",  # ru <-> uk
    "Helsinki-NLP/opus-mt-zlw-zlw",  # pl interne Übersetzung
    "Helsinki-NLP/opus-mt-gmw-gmw",  # en/de/nl untereinander
    "Helsinki-NLP/opus-mt-gmq-gmq",  # sv untereinander
    "Helsinki-NLP/opus-mt-itc-itc",  # fr/es/it/pt untereinander
    # Tatoeba-Modelle (transformer-align Architektur)
    "Helsinki-NLP/opus-tatoeba-en-ja",
    "Helsinki-NLP/opus-tatoeba-fr-it",
    "Helsinki-NLP/opus-tatoeba-es-zh",
    "Helsinki-NLP/opus-tatoeba-en-tr",
]

# Duplikate entfernen, Reihenfolge beibehalten
seen = set()
MODELS = []
for m in ALL_MODELS:
    if m not in seen:
        seen.add(m)
        MODELS.append(m)

# Alle Modelle laden, die von TalkBridge genutzt werden
# def download_used_models(output_dir: Path):
#     try:
#         from huggingface_hub import snapshot_download
#     except ImportError:
#         print("Fehler: huggingface_hub nicht installiert.")
#         print("Bitte ausführen: pip install huggingface_hub")
#         sys.exit(1)
        
#     models_to_download = USED_MODELS
    
#     print(f"Zielordner: {output_dir.resolve()}")
#     print(f"Modelle gesamt: {len(models_to_download)}")
#     print()
    
#     output_dir.mkdir(parents=True, exist_ok=True)

#     failed = []
#     for i, model_id in enumerate(models_to_download, 1):
#         local_name = model_id.replace("/", "__")
#         local_path = output_dir / local_name

#         if local_path.exists() and any(local_path.iterdir()):
#             print(f"[{i}/{len(models_to_download)}] Übersprungen (existiert): {model_id}")
#             continue

#         print(f"[{i}/{len(models_to_download)}] Lade: {model_id}")
#         try:
#             snapshot_download(
#                 repo_id=model_id,
#                 local_dir=str(local_path),
#                 ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
#             )
#             print(f"  -> Gespeichert: {local_path}")
#         except Exception as e:
#             print(f"  -> FEHLER: {e}")
#             failed.append((model_id, str(e)))

#     print()
#     print(f"Fertig. {len(models_to_download) - len(failed)}/{len(models_to_download)} Modelle erfolgreich.")
#     if failed:
#         print(f"\nFehlgeschlagene Modelle ({len(failed)}):")
#         for model_id, err in failed:
#             print(f"  {model_id}: {err}")
    

def download_models(output_dir: Path, only_new: bool = False, dry_run: bool  = False, ignore_new: bool = False, model_name: str = "", contains: str = ""):
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Fehler: huggingface_hub nicht installiert.")
        print("Bitte ausführen: pip install huggingface_hub")
        sys.exit(1)

    models_to_download = MODELS
    if only_new:
        models_to_download = [m for m in MODELS if "tc-big" in m or "tc-base" in m]
        print(f"--only-new: {len(models_to_download)} tc-big / tc-base Modelle ausgewählt")
    if ignore_new:
        models_to_download = [m for m in MODELS if "tc-big" not in m and "tc-base" not in m]
    if contains != "":
        models_to_download = [m for m in MODELS if contains in m]
    if model_name != "":
        models_to_download = [m for m in MODELS if model_name in m]

    print(f"Zielordner: {output_dir.resolve()}")
    print(f"Modelle gesamt: {len(models_to_download)}")
    print()

    if dry_run:
        print("=== DRY RUN – nichts wird heruntergeladen ===")
        for m in models_to_download:
            local_name = m.replace("/", "__")
            print(f"  {m}")
            print(f"    -> {output_dir / local_name}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    failed = []
    for i, model_id in enumerate(models_to_download, 1):
        local_name = model_id.replace("/", "__")
        local_path = output_dir / local_name

        if local_path.exists() and any(local_path.iterdir()):
            print(f"[{i}/{len(models_to_download)}] Übersprungen (existiert): {model_id}")
            continue

        print(f"[{i}/{len(models_to_download)}] Lade: {model_id}")
        try:
            snapshot_download(
                repo_id=model_id,
                local_dir=str(local_path),
                ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
            )
            print(f"  -> Gespeichert: {local_path}")
        except Exception as e:
            print(f"  -> FEHLER: {e}")
            failed.append((model_id, str(e)))

    print()
    print(f"Fertig. {len(models_to_download) - len(failed)}/{len(models_to_download)} Modelle erfolgreich.")
    if failed:
        print(f"\nFehlgeschlagene Modelle ({len(failed)}):")
        for model_id, err in failed:
            print(f"  {model_id}: {err}")


def main():
    parser = argparse.ArgumentParser(description="OPUS-MT Modelle herunterladen")
    parser.add_argument(
        "--output", "-o",
        default="./opus_mt_models",
        help="Zielordner für die Modelle (Standard: ./opus_mt_models)"
    )
    parser.add_argument(
        "--only-new",
        action="store_true",
        help="Nur tc-big und tc-base Modelle herunterladen"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Nur anzeigen was geladen würde, ohne tatsächlich zu laden"
    )
    parser.add_argument(
        "--ignore-new",
        action="store_true",
        help="Alles außer tc-big und tc-base wird geladen"
    )
    parser.add_argument(
        "--model-name",
        default="",
        # choices=ALL_MODELS,
        help="Download only one specific model"
    )
    parser.add_argument(
        "--contains",
        default="",
        help="only download models with ___ in its name"
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    download_models(output_dir, args.only_new, args.dry_run, args.ignore_new, args.model_name, args.contains)
    # download_used_models(output_dir)


if __name__ == "__main__":
    main()
