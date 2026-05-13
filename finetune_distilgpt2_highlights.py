# finetune_distilgpt2_highlights.py
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    DataCollatorForLanguageModeling, Trainer, TrainingArguments
)

model_name = "distilgpt2"
out_dir    = "./ft_distilgpt2_highlights"

tok = AutoTokenizer.from_pretrained(model_name)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# Use short texts to avoid 512-token issues
train = load_dataset("cnn_dailymail", "3.0.0", split="train[:20000]")
def prep(ex):
    return tok(ex["highlights"], truncation=True, max_length=256)
train = train.map(prep, batched=True, remove_columns=train.column_names)

model = AutoModelForCausalLM.from_pretrained(model_name)

args = TrainingArguments(
    output_dir=out_dir,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    num_train_epochs=1,
    learning_rate=5e-5,
    weight_decay=0.01,
    logging_steps=100,
    save_steps=2000,
    save_total_limit=1,
    report_to=[],
)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train,
    data_collator=DataCollatorForLanguageModeling(tok, mlm=False),
)
trainer.train()

trainer.save_model(out_dir)
tok.save_pretrained(out_dir)
print("Saved fine-tuned model to", out_dir)
