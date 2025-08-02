import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import (
    DataCollatorForSeq2Seq, AutoTokenizer, AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments, Trainer, Seq2SeqTrainer, EarlyStoppingCallback
)


class T5Generator:
    def __init__(self, model_checkpoint):
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
        self.data_collator = DataCollatorForSeq2Seq(self.tokenizer)
        self.device = 'cuda' if torch.has_cuda else ('mps' if torch.has_mps else 'cpu')

    def tokenize_function_inputs(self, sample):
        """
        Udf to tokenize the input dataset.
        """
        model_inputs = self.tokenizer(sample['text'], max_length=512, truncation=True)
        labels = self.tokenizer(sample["labels"], max_length=64, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
        
    def train(self, tokenized_datasets, **kwargs):
        """
        Train the generative model.
        """
        #Set training arguments
        args = Seq2SeqTrainingArguments(
            **kwargs
        )

        # Define trainer object
        trainer = Seq2SeqTrainer(
            self.model,
            args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"] if tokenized_datasets.get("validation") is not None else None,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=15)]
        )
        print("Trainer device:", trainer.args.device)

        # Finetune the model
        torch.cuda.empty_cache()
        print('\nModel training started ....')
        trainer.train()

        # Save best model
        trainer.save_model()
        return trainer

    def get_labels(self, tokenized_dataset, batch_size = 4, max_length = 128, sample_set = 'train'):
        """
        Get the predictions from the trained model.
        """
        def collate_fn(batch):
            input_ids = [torch.tensor(example['input_ids']) for example in batch]
            input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            return input_ids
        
        dataloader = DataLoader(tokenized_dataset[sample_set], batch_size=batch_size, collate_fn=collate_fn)
        predicted_output = []
        self.model.to(self.device)
        print('Model loaded to: ', self.device)

        for batch in tqdm(dataloader):
            batch = batch.to(self.device)
            output_ids = self.model.generate(batch, max_length = max_length)
            output_texts = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            for output_text in output_texts:
                predicted_output.append(output_text)
        return predicted_output
    
    def get_metrics(self, y_true, y_pred, is_triplet_extraction=False):
        total_pred = 0
        total_gt = 0
        tp = 0
        if not is_triplet_extraction:
            for gt, pred in zip(y_true, y_pred):
                gt_list = gt.split(', ')
                pred_list = pred.split(', ')
                total_pred+=len(pred_list)
                total_gt+=len(gt_list)
                for gt_val in gt_list:
                    for pred_val in pred_list:
                        if pred_val in gt_val or gt_val in pred_val:
                            tp+=1
                            break

        else:
            for gt, pred in zip(y_true, y_pred):
                gt_list = gt.split(', ')
                pred_list = pred.split(', ')
                total_pred+=len(pred_list)
                total_gt+=len(gt_list)
                for gt_val in gt_list:
                    gt_asp = gt_val.split(':')[0]

                    try:
                        gt_op = gt_val.split(':')[1]
                    except:
                        continue

                    try:
                        gt_sent = gt_val.split(':')[2]
                    except:
                        continue

                    for pred_val in pred_list:
                        pr_asp = pred_val.split(':')[0]

                        try:
                            pr_op = pred_val.split(':')[1]
                        except:
                            continue

                        try:
                            pr_sent = gt_val.split(':')[2]
                        except:
                            continue

                        if pr_asp in gt_asp and pr_op in gt_op and gt_sent == pr_sent:
                            tp+=1

        p = tp/total_pred
        r = tp/total_gt
        return p, r, 2*p*r/(p+r), None
    from sklearn.metrics import precision_score, recall_score, f1_score

    def get_metrics2(self, y_true, y_pred, is_triplet_extraction=False):
        """
        Menghitung micro, macro, dan weighted precision, recall, F1-score
        untuk sentiment classification (positive, neutral, negative).
        """
        true_labels = []
        pred_labels = []
        
        # Daftar label sentimen yang digunakan
        sentiment_classes = ["positive", "neutral", "negative"]

        if not is_triplet_extraction:
            for gt, pred in zip(y_true, y_pred):
                gt_list = gt.split(', ')
                pred_list = pred.split(', ')
                
                for gt_val in gt_list:
                    if ":" in gt_val:  # Pastikan format "aspek:sentimen"
                        gt_aspect, gt_sentiment = gt_val.split(":")
                        true_labels.append(gt_sentiment)
                        
                        # Ambil prediksi yang sesuai atau default ke "neutral"
                        pred_sentiment = next((p.split(":")[1] for p in pred_list if p.startswith(gt_aspect + ":")), "neutral")
                        pred_labels.append(pred_sentiment)

        else:
            for gt, pred in zip(y_true, y_pred):
                gt_list = gt.split(', ')
                pred_list = pred.split(', ')

                for gt_val in gt_list:
                    parts = gt_val.split(":")
                    if len(parts) < 3:
                        continue  # Skip jika format salah
                    
                    gt_aspect, gt_opinion, gt_sentiment = parts
                    true_labels.append(gt_sentiment)

                    pred_sentiment = next(
                        (p.split(":")[2] for p in pred_list if p.startswith(gt_aspect + ":" + gt_opinion)), 
                        "neutral"
                    )
                    pred_labels.append(pred_sentiment)

        # Handle jika tidak ada data (hindari error)
        if not true_labels or not pred_labels:
            return {
                "micro": {"precision": 0, "recall": 0, "f1": 0},
                "macro": {"precision": 0, "recall": 0, "f1": 0},
                "weighted": {"precision": 0, "recall": 0, "f1": 0}
            }

        # Hitung Micro, Macro, dan Weighted Metrics
        precision_micro = precision_score(true_labels, pred_labels, labels=sentiment_classes, average='micro', zero_division=0)
        recall_micro = recall_score(true_labels, pred_labels, labels=sentiment_classes, average='micro', zero_division=0)
        f1_micro = f1_score(true_labels, pred_labels, labels=sentiment_classes, average='micro', zero_division=0)

        precision_macro = precision_score(true_labels, pred_labels, labels=sentiment_classes, average='macro', zero_division=0)
        recall_macro = recall_score(true_labels, pred_labels, labels=sentiment_classes, average='macro', zero_division=0)
        f1_macro = f1_score(true_labels, pred_labels, labels=sentiment_classes, average='macro', zero_division=0)

        precision_weighted = precision_score(true_labels, pred_labels, labels=sentiment_classes, average='weighted', zero_division=0)
        recall_weighted = recall_score(true_labels, pred_labels, labels=sentiment_classes, average='weighted', zero_division=0)
        f1_weighted = f1_score(true_labels, pred_labels, labels=sentiment_classes, average='weighted', zero_division=0)

        # Return hasil dalam format dictionary
        return {
            "micro": {"precision": precision_micro, "recall": recall_micro, "f1": f1_micro},
            "macro": {"precision": precision_macro, "recall": recall_macro, "f1": f1_macro},
            "weighted": {"precision": precision_weighted, "recall": recall_weighted, "f1": f1_weighted}
        }

    # Contoh Penggunaan
    # train_metrics = get_metrics(id_tr_labels, id_tr_pred_labels)
    # print("Train Metrics:", train_metrics)



class T5Classifier:
    def __init__(self, model_checkpoint):
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, force_download = True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint, force_download = True)
        self.data_collator = DataCollatorForSeq2Seq(self.tokenizer)
        self.device = 'cuda' if torch.has_cuda else ('mps' if torch.has_mps else 'cpu')

    def tokenize_function_inputs(self, sample):
        """
        Udf to tokenize the input dataset.
        """
        sample['input_ids'] = self.tokenizer(sample["text"], max_length = 512, truncation = True).input_ids
        sample['labels'] = self.tokenizer(sample["labels"], max_length = 64, truncation = True).input_ids
        return sample
        
    def train(self, tokenized_datasets, **kwargs):
        """
        Train the generative model.
        """

        # Set training arguments
        args = Seq2SeqTrainingArguments(
            **kwargs
            )

        # Define trainer object
        trainer = Trainer(
            self.model,
            args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"] if tokenized_datasets.get("validation") is not None else None,
            tokenizer=self.tokenizer, 
            data_collator = self.data_collator 
        )
        print("Trainer device:", trainer.args.device)

        # Finetune the model
        torch.cuda.empty_cache()
        print('\nModel training started ....')
        trainer.train()

        # Save best model
        trainer.save_model()
        return trainer

    def get_labels(self, tokenized_dataset, batch_size = 4, sample_set = 'train'):
        """
        Get the predictions from the trained model.
        """
        def collate_fn(batch):
            input_ids = [torch.tensor(example['input_ids']) for example in batch]
            input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            return input_ids
        
        dataloader = DataLoader(tokenized_dataset[sample_set], batch_size=batch_size, collate_fn=collate_fn)
        predicted_output = []
        self.model.to(self.device)
        print('Model loaded to: ', self.device)

        for batch in tqdm(dataloader):
            batch = batch.to(self.device)
            output_ids = self.model.generate(batch)
            output_texts = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            for output_text in output_texts:
                predicted_output.append(output_text)
        return predicted_output
    
    def get_metrics(self, y_true, y_pred):
        return precision_score(y_true, y_pred, average='macro'), recall_score(y_true, y_pred, average='macro'), \
            f1_score(y_true, y_pred, average='macro'), accuracy_score(y_true, y_pred)