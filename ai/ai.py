import pandas as pd
import torch
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import numpy as np

# === PARAMETRI ===
MODEL_NAME = "dbmdz/bert-base-italian-uncased"
CSV_PATH = "dataset_griglia_1200_frasi_finale.csv"  # Sostituire se il file ha altro nome
LABEL_COLUMNS = [
    "fisica_maschile",
    "stereotipi_ruoli",
    "assenza_modelli_femminili",
    "marginalizzazione_donne",
    "confini_ruolo_scientifico",
    "assenza_agency_femminile"
]
TEXT_COLUMN = "frase"

# === 1. CARICA DATASET ===
df = pd.read_csv(CSV_PATH)
df = df.dropna(subset=[TEXT_COLUMN])  # Rimuovi righe vuote
df[LABEL_COLUMNS] = df[LABEL_COLUMNS].astype(int)  # Assicurati che siano interi

# === 2. PREPARA DATASET HF ===
dataset = Dataset.from_pandas(df[[TEXT_COLUMN] + LABEL_COLUMNS])
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(batch[TEXT_COLUMN], padding="max_length", truncation=True, max_length=128)

dataset = dataset.map(tokenize, batched=True)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask"] + LABEL_COLUMNS)

# === 3. DIVIDI IN TRAIN/TEST ===
dataset = dataset.train_test_split(test_size=0.2)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

# === 4. MODELLO BERT MULTILABEL ===
model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(LABEL_COLUMNS),
    problem_type="multi_label_classification"
)

# === 5. FUNZIONE DI VALUTAZIONE ===
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = (torch.sigmoid(torch.tensor(logits)) > 0.5).int().numpy()
    labels = labels.astype(int)
    accuracy = (preds == labels).mean()
    return {"accuracy": accuracy}

# === 6. TRAINING ===
training_args = TrainingArguments(
    output_dir="./results_griglia_ai",
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=4,
    save_steps=500,
    logging_dir="./logs",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()

# === 7. SALVA IL MODELLO ADDDESTRATO ===
trainer.save_model("modello_griglia_ai")
tokenizer.save_pretrained("modello_griglia_ai")

print("✅ Modello addestrato e salvato nella cartella 'modello_griglia_ai'")
