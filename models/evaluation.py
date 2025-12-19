import pandas as pd
import evaluate

def evaluate_function(texts_model: list, texts_target: list) -> list:
    bleu_score = evaluate.load("bleu")

    bleu_final = bleu_score.compute(predictions=texts_model, references=texts_target)
    return bleu_final["bleu"]

file_name = "statistics\\translation_deepseek.txt"
with open(file_name, "rt", encoding="utf-8") as file:
    texts_model = [line.replace("\n", "") for line in file]

df = pd.read_csv("datasets\\giga_chat_dataset.csv")

def strip_function(text: str) -> str:
    return text.strip()

texts_target = df["glosses"].apply(strip_function).tolist()

print(evaluate_function(texts_model=texts_model, texts_target=texts_target))