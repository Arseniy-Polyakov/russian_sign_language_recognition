import pandas as pd

def parsing_glosses() -> list:
    df = pd.read_csv("datasets\\annotations.csv", sep="\t")
    glosses_slovo = [item.upper().replace(",", "").strip() for item in df["text"].tolist()]
    glosess_slovo_unique = list(set(glosses_slovo))
    return glosess_slovo_unique

def parsing_texts() -> list:
    df = pd.read_csv("datasets\\giga_chat_dataset.csv")
    russian_texts = df["russian"].tolist()
    return russian_texts