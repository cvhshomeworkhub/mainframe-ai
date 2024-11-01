import time
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

class ChangeHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.src_path.endswith("data.txt"):
            print("Detected changes in data.txt. Retraining model...")
            retrain_model()

def load_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return [{'text': line.strip()} for line in lines if line.strip()]

def retrain_model():
    dataset = load_data('data.txt')
    dataset = Dataset.from_list(dataset)

    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    tokenized_data = dataset.map(lambda x: tokenizer(x['text'], truncation=True, padding=True), batched=True)

    model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy='epoch',
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        num_train_epochs=1,
        logging_dir='./logs',
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data,
    )

    trainer.train()
    print("Model retrained successfully.")

def main():
    path = "."
    event_handler = ChangeHandler()
    observer = Observer()
    observer.schedule(event_handler, path, recursive=False)
    observer.start()

    try:
        print("Monitoring data.txt for changes...")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    main()
