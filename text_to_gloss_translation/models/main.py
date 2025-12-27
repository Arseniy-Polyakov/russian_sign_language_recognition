import pandas as pd
from openai import OpenAI
from huggingface_hub import InferenceClient

from config import AUTHORIZATION_ID, HF_TOKEN
from parsing_data import parsing_glosses, parsing_texts
from giga_chat import (getting_token_gigachat, 
                       giga_call,
                       monitoring_sber)

system_prompt = f"""Ты переводчик с русского вербального языка на язык глоссов русского жестового языка.
Глоссы - это значения жестов русского жестового языка.
Список существующих глоссов русского жестового языка, используй только их для перевода: {parsing_glosses()}. 

Твоя задача - перевести предложения с русского вербального языка на глоссы русского жестового языка. 
В ответе ты должен использовать только предоставленные глоссы, никакие другие слова глоссы использовать нельзя.
Используй только глоссы в той грамматической форме, в которой они есть в списке, не меняй форму глоссов.
Используй верхний регистр для записи глоссов.
Пиши каждое новое предложение с новой строки, используй нумерованный список в ответе.
Не пиши комментарии и дополнительную информацию в ответе.

Примеры формата вывода ответа: 
Тексты на русском вербальном языке:
Испуганная девочка съёжилась от страха
Кролики любят лакомиться
Огромный медведь был на солнце после зимней спячки
Цветовая гамма весеннего пейзажа радует глаз
Купи питьевой воды

Перевод на глоссы:
1. ИСПУГАННЫЙ ДЕВОЧКА ИСПЫТЫВАТЬ СТРАХ
2. КРОЛИК ЛЮБИТЬ ЕСТЬ 
3. БОЛЬШОЙ МЕДВЕДЬ БЫТЬ СОЛНЦЕ 
4. ЦВЕТОВОЙ ОТТЕНОК ВЕСНА ПРИЯТНЫЙ ГЛАЗ
5. КУПИТЬ ПИТЬЕВАЯ ВОДА
"""
user_prompt = f"Тексты для перевода: {parsing_texts()[:50]}"

# system_prompt = "Переведи на немецкий язык"
# user_prompt = "Жест"

def giga_chat(system_prompt: str, user_prompt: str):
    model = "GigaChat"
    ACCESS_TOKEN = getting_token_gigachat()
    giga_answer = giga_call(model=model, system_prompt=system_prompt, user_prompt=user_prompt, TOKEN=ACCESS_TOKEN)
    monitoring_check = monitoring_sber(model=model, system_prompt=system_prompt, user_prompt=user_prompt, answer_models=giga_answer)
    print("ОТВЕТ ОТ МОДЕЛИ", giga_answer)
    print("----------------------------")
    print(monitoring_check)

def hugging_face_model(system_prompt: str, user_prompt: str, HF_TOKEN: str) -> str:
    client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=HF_TOKEN
    )
    completion = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-V3-0324", 
        messages=[
            {"role": "system", "content": system_prompt}, 
            {"role": "user", "content": user_prompt}
        ]
    )
        
    tokens = completion.usage.total_tokens
    answer = completion.choices[0].message.content
    print("ОТВЕТ ОТ МОДЕЛИ", answer)
    print("----------------------------")
    print("КОЛИЧЕСТВО ЗАДЕЙСТВОВАННЫХ ТОКЕНОВ HUGGING FACE: ", tokens)

# def hugging_face_hub(system_prompt: str, user_prompt: str):
#     messages = [
#         {"role": "system", "content": system_prompt}, 
#         {"role": "user", "content": user_prompt}
#     ]
#     client = InferenceClient(
#         messages=messages,
#         model="microsoft/Phi-3-mini-4k-instruct", 
#         provider="", 
#         api_key=HF_TOKEN
#     )
#     answer = client.chat_completion(messages=messages, max_tokens=1000)
#     return answer

print(hugging_face_model(system_prompt=system_prompt, user_prompt=user_prompt, HF_TOKEN=HF_TOKEN))
