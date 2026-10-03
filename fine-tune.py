from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)
import evaluate


# -------------------------
# 1. Configuration
# -------------------------

checkpoint = "bert-base-uncased"

# -------------------------
# 2. Load dataset
# -------------------------

raw_datasets = load_dataset("nyu-mll/glue", "mrpc")

# -------------------------
# 3. Tokenizer
# -------------------------

tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def tokenize_function(examples):
    return tokenizer(
        examples["sentence1"],
        examples["sentence2"],
        truncation=True,
    )


tokenized_datasets = raw_datasets.map(
    tokenize_function,
    batched=True,
    remove_columns=["sentence1", "sentence2", "idx"],
)

# -------------------------
# 4. Dynamic padding
# -------------------------

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer
)

# -------------------------
# 5. Model
# -------------------------

model = AutoModelForSequenceClassification.from_pretrained(
    checkpoint,
    num_labels=2,
)

# -------------------------
# 6. Evaluation metrics
# -------------------------

accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    predictions = predictions.argmax(axis=-1)

    accuracy_score = accuracy.compute(
        predictions=predictions,
        references=labels,
    )

    f1_score = f1.compute(
        predictions=predictions,
        references=labels,
    )

    return {
        "accuracy": accuracy_score["accuracy"],
        "f1": f1_score["f1"],
    }


# -------------------------
# 7. Training configuration
# -------------------------

training_args = TrainingArguments(
    output_dir="./bert-mrpc",
    
    # Training
    num_train_epochs=3,
    learning_rate=2e-5,
    weight_decay=0.01,
    
    # Batch sizes
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    
    # Evaluation
    eval_strategy="epoch",
    save_strategy="epoch",
    
    # Keep the best checkpoint
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    
    # Logging
    logging_strategy="steps",
    logging_steps=50,
    
    # Prevent excessive checkpoint storage
    save_total_limit=2,
    
    # Don't send data to external logging services
    report_to="none",
)

# -------------------------
# 8. Trainer
# -------------------------

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# -------------------------
# 9. Train
# -------------------------

trainer.train()

# -------------------------
# 10. Final evaluation
# -------------------------

results = trainer.evaluate()

print("\nEvaluation results:")
for key, value in results.items():
    print(f"{key}: {value}")
