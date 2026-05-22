import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ["ARGOS_PACKAGES_DIR"] = os.path.join(BASE_DIR, "argos", "models")

import argostranslate.package

# Liste der Sprachen
languages = {
    "en": "English",
    "de": "German",
    "fr": "French",
    # "zh": "Chinese",
    "ru": "Russian",
    "uk": "Ukrainian",
    "es": "Spanish",
    "pt": "Portuguese",
    "tr": "Turkish",
    "it": "Italian",
    # "ja": "Japanese",
    "sv": "Swedish",
    "pl": "Polish",
    # "ko": "Korean",
    "nl": "Dutch",
    # "vi": "Vietnamese",
}

# 1. Update Index
argostranslate.package.update_package_index()
available_packages = argostranslate.package.get_available_packages()


for from_code in languages:
    for to_code in languages:
        if from_code == to_code:
            continue  # gleiche Sprache überspringen

        # Prüfen, ob Paket existiert
        package_to_install = next(
            (p for p in available_packages if p.from_code == from_code and p.to_code == to_code),
            None
        )

        if package_to_install:
            print(f"Installiere {from_code} -> {to_code} ...")
            argostranslate.package.install_from_path(package_to_install.download())
        else:
            print(f"Kein Paket für {from_code} -> {to_code} verfügbar.")
            
installed = argostranslate.package.get_installed_packages()
for pkg in installed:
    print(f"installed: {pkg}")