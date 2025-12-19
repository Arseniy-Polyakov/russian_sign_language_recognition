from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from transformers import (
    PreTrainedTokenizerFast, 
    BartConfig, 
    BartForConditionalGeneration, 
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer
)

from datasets import load_dataset

dataset = load_dataset(
    "csv", 
    data_files = {
        "train": "datasets/train.csv", 
        "test": "datasets/test.csv"
    }, 
    delimiter=",", 
    column_names=["russian", "rsl"]
)

tokenizer = Tokenizer(models.BPE())
pre_tokenizer = pre_tokenizers.ByteLevel()
trainer = trainers.BpeTrainer(vocab_size=100, special_tokens=[
    "<pad>", "<unk>", "<s>", "</s>"
])

all_texts = []
for item in dataset["train"]:
    all_texts.append(item["russian"])
    all_texts.append(item["rsl"])

tokenizer.train_from_iterator(all_texts, trainer)
# tokenizer.save("tokenizer.json")

hf_tokenizer = PreTrainedTokenizerFast(tokenizer_file="tokenizer.json")
hf_tokenizer.pad_token = "<pad>"
hf_tokenizer.eos_token = "</s>"
hf_tokenizer.bos_token = "<s>"

config = BartConfig(
    vocab_size=hf_tokenizer.vocab_size,
    d_model=512,
    encoder_layers=6,
    decoder_layers=6,
    encoder_attention_heads=8,
    decoder_attention_heads=8,
    decoder_ffn_dim=2048,
    encoder_ffn_dim=2048,
    max_position_embeddings=512
)

model = BartForConditionalGeneration(config)
collator = DataCollatorForSeq2Seq(hf_tokenizer, model=model)

def tokenize(batch):
    model_inputs = hf_tokenizer(
        batch["russian"],
        truncation=True,
        padding="max_length",
        max_length=128
    )
    labels = hf_tokenizer(
        batch["rsl"],
        truncation=True,
        padding="max_length",
        max_length=128
    )["input_ids"]

    model_inputs["labels"] = labels
    return model_inputs

tokenized = dataset.map(tokenize, batched=True)

training_args = Seq2SeqTrainingArguments(
    output_dir="transformer",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,
    warmup_steps=1000,
    max_steps=20000,
    learning_rate=5e-4,
    logging_steps=100,
    save_steps=2000,
    save_total_limit=3,
    fp16=False,                 
    predict_with_generate=True
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=collator,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    tokenizer=hf_tokenizer
)

trainer.train()
