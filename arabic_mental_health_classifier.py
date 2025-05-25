# Arabic Mental Health Classification using AraBERT
# This script fine-tunes AraBERT for mental health classification on social media posts

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import re
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class ArabicMentalHealthDataset(Dataset):
    """Custom dataset class for Arabic mental health classification"""
    def __init__(self, texts, labels, tokenizer, max_length=128):
        """
        Args:
            texts: List of Arabic text samples
            labels: List of corresponding labels
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Tokenize and encode the text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def preprocess_arabic_text(text):
    """Preprocess Arabic text for classification"""
    if pd.isna(text):
        return ''
        
    text = str(text)
    
    # Normalize Arabic text
    text = re.sub(r'[إأآ]', 'ا', text)  # Normalize different forms of 'a'
    text = re.sub(r'[ؤئ]', 'ي', text)  # Normalize different forms of 'y'
    
    # Remove diacritical marks
    text = re.sub(r'[ًٌٍَُِّْ]', '', text)
    
    # Remove special characters
    text = re.sub(r'[\u061B\u061F\u066A-\u066D\u06D4]', '', text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Remove mentions and hashtags
    text = re.sub(r'@[\w\u0621-\u064A]+', '', text)
    text = re.sub(r'#\w+', '', text)
    
    # Remove emojis
    text = re.sub(r'[\U0001F600-\U0001F64F]', '', text)
    text = re.sub(r'[\U0001F300-\U0001F5FF]', '', text)
    text = re.sub(r'[\U0001F680-\U0001F6FF]', '', text)
    text = re.sub(r'[\U0001F1E0-\U0001F1FF]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def prepare_dataset(csv_path, text_column, label_column):
    """Prepare dataset for training"""
    print("\nPreparing dataset...")
    try:
        # Load dataset
        df = pd.read_csv(csv_path)
        print(f"\nOriginal dataset shape: {df.shape}")
        
        # Display label distribution
        print("\nLabel distribution:")
        print(df[label_column].value_counts())
        
        # Preprocess text
        print("\nPreprocessing Arabic text...")
        df[text_column] = df[text_column].apply(preprocess_arabic_text)
        
        # Remove empty texts
        df = df[df[text_column].str.len() > 0]
        print(f"\nDataset shape after preprocessing: {df.shape}")
        
        # Encode labels
        label_encoder = LabelEncoder()
        df['encoded_labels'] = label_encoder.fit_transform(df[label_column])
        
        # Display label mapping
        label_mapping = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))
        print("\nLabel Mapping:")
        for label, idx in label_mapping.items():
            print(f"{idx}: {label}")
        
        return df, label_encoder
        
    except Exception as e:
        print(f"Error preparing dataset: {str(e)}")
        raise

def setup_model_and_trainer(model_name="aubmindlab/bert-base-arabertv2", num_labels=None):
    """Set up model, tokenizer, and trainer"""
    print("\nSetting up model and trainer...")
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_info()  # Enable informative logs
    import os
    import requests
    
    # Set cache directory explicitly to avoid permission issues
    cache_dir = './model_cache'
    os.makedirs(cache_dir, exist_ok=True)
    print(f"Using cache directory: {cache_dir}")
    
    print(f"\nDownloading tokenizer for {model_name}...")
    try:
        # Use a short timeout for initial connection to verify server is reachable
        session = requests.Session()
        session.mount('https://', requests.adapters.HTTPAdapter(max_retries=3))
        
        # Try with a smaller model first if specified model has issues
        alternative_model = "asafaya/bert-base-arabic" if model_name == "aubmindlab/bert-base-arabertv2" else model_name
        print(f"Will try alternative model {alternative_model} if main model fails")
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,  # Use faster tokenizer if available
            cache_dir=cache_dir,
            local_files_only=False,  # Force online check first
        )
        print("Tokenizer loaded successfully!")
    except Exception as e:
        print(f"Error loading primary tokenizer: {str(e)}")
        print("Trying alternative model tokenizer...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                alternative_model,
                use_fast=True,
                cache_dir=cache_dir,
            )
            print(f"Alternative tokenizer loaded successfully!")
        except Exception as e2:
            print(f"Error loading alternative tokenizer: {str(e2)}")
            raise
    
    print(f"\nDownloading pre-trained model {model_name}...")
    print("This may take a few minutes - please be patient...")
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
            cache_dir=cache_dir,
            local_files_only=False,  # Force online check first
        )
        model = model.to(device)
        print(f"Model loaded successfully and moved to {device}!")
    except Exception as e:
        print(f"Error loading primary model: {str(e)}")
        print("Trying alternative model...")
        try:
            model = AutoModelForSequenceClassification.from_pretrained(
                alternative_model,
                num_labels=num_labels,
                ignore_mismatched_sizes=True,
                cache_dir=cache_dir,
            )
            model = model.to(device)
            print(f"Alternative model loaded successfully and moved to {device}!")
        except Exception as e2:
            print(f"Error loading alternative model: {str(e2)}")
            raise
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=100,
        eval_steps=500,
        save_steps=500,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        report_to=None,  # Disable wandb for simpler setup
        fp16=True if torch.cuda.is_available() else False,
        dataloader_num_workers=2
    )
    
    # Set evaluation and save strategies
    training_args.evaluation_strategy = "steps"  # Evaluate at specified eval_steps
    training_args.save_strategy = "steps"  # Save checkpoints at specified save_steps
    training_args.load_best_model_at_end = True  # Load the best model based on evaluation  
    
    return model, tokenizer, training_args

def compute_metrics(eval_pred):
    """Compute evaluation metrics"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = accuracy_score(labels, predictions)
    return {"accuracy": accuracy}

def train_and_evaluate(
    csv_path,
    text_column="tweet",
    label_column="category",
    test_size=0.2,
    random_state=42,
    max_length=128  # Added parameter for flexibility
):
    """Main training and evaluation pipeline"""
    try:
        # Check if file exists
        import os
        if not os.path.exists(csv_path):
            print(f"Dataset file not found: {csv_path}")
            print("Make sure the path is correct and the file exists.")
            return None, None, None
        
        print(f"\nProcessing dataset from: {csv_path}")
        print(f"Looking for columns: {text_column} and {label_column}")
        
        # Prepare dataset with progress updates
        print("\n[1/7] Loading and preparing dataset...")
        df, label_encoder = prepare_dataset(csv_path, text_column, label_column)
        
        # Split dataset
        print("\n[2/7] Splitting dataset into train and test sets...")
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=df['encoded_labels']
        )
        print(f"Train set: {len(train_df)} samples, Test set: {len(test_df)} samples")
        
        # Get number of labels
        num_labels = len(label_encoder.classes_)
        print(f"\nDetected {num_labels} unique labels.")
        
        # Set up model and trainer
        print("\n[3/7] Setting up model and trainer...")
        model, tokenizer, training_args = setup_model_and_trainer(
            model_name="aubmindlab/bert-base-arabertv2",
            num_labels=num_labels
        )
        
        # Create datasets with progress updates
        print("\n[4/7] Creating tokenized datasets...")
        print("Tokenizing training data...")
        train_dataset = ArabicMentalHealthDataset(
            train_df[text_column].tolist(),
            train_df['encoded_labels'].tolist(),
            tokenizer,
            max_length=max_length
        )
        
        print("Tokenizing test data...")
        test_dataset = ArabicMentalHealthDataset(
            test_df[text_column].tolist(),
            test_df['encoded_labels'].tolist(),
            tokenizer,
            max_length=max_length
        )
        print(f"Datasets created with max sequence length of {max_length}")
        
        # Create trainer with progress updates
        print("\n[5/7] Creating trainer...")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )
        
        # Train model with progress updates
        print("\n[6/7] Starting training process...")
        print("This may take some time. Training progress will be displayed below:")
        
        # Add a timestamp to monitor progress
        import time
        start_time = time.time()
        
        # Train the model
        train_result = trainer.train()
        
        # Calculate training time
        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
        print(f"Training metrics: {train_result.metrics}")
        
        # Evaluate model
        print("\n[7/7] Evaluating model...")
        eval_results = trainer.evaluate()
        print(f"\nEvaluation results: {eval_results}")
        
        # Get predictions
        print("\nGenerating predictions on test set...")
        predictions = trainer.predict(test_dataset)
        y_pred = np.argmax(predictions.predictions, axis=1)
        y_true = predictions.label_ids
        
        # Convert back to original labels
        print("Converting predictions to original labels...")
        y_true_labels = label_encoder.inverse_transform(y_true)
        y_pred_labels = label_encoder.inverse_transform(y_pred)
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_true_labels, y_pred_labels))
        
        # Plot confusion matrix
        cm = confusion_matrix(y_true_labels, y_pred_labels)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=label_encoder.classes_,
                   yticklabels=label_encoder.classes_)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        plt.close()
        
        return model, tokenizer, label_encoder
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise

def predict_text(model, tokenizer, label_encoder, text):
    """Predict mental health category for a single text"""
    try:
        # Preprocess text
        text = preprocess_arabic_text(text)
        
        # Tokenize
        inputs = tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors='pt'
        ).to(device)
        
        # Get prediction
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            prediction = torch.argmax(logits, dim=1).item()
            
        # Convert to label
        predicted_label = label_encoder.inverse_transform([prediction])[0]
        confidence = torch.softmax(logits, dim=1).max().item()
        
        return predicted_label, confidence
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return None, None

# Training arguments with smaller batch size and faster processing
def get_training_args(output_dir="./results", num_epochs=2, batch_size=8):
    """Create training arguments optimized for faster processing"""
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,  # Reduced epochs for faster training
        per_device_train_batch_size=batch_size,  # Smaller batch size to reduce memory usage
        per_device_eval_batch_size=batch_size * 2,
        warmup_steps=100,  # Reduced warmup steps
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,  # More frequent logging to see progress
        eval_steps=100,  # More frequent evaluation
        save_steps=100,  # More frequent saving
        metric_for_best_model="accuracy",
        greater_is_better=True,
        report_to=None,  # Disable wandb for simpler setup
        fp16=True if torch.cuda.is_available() else False,
        dataloader_num_workers=0,  # Avoid multiprocessing issues
        disable_tqdm=False,  # Show progress bars
        logging_first_step=True,  # Log the first training step
    )
    
    # Set evaluation and save strategies
    training_args.evaluation_strategy = "steps"
    training_args.save_strategy = "steps"
    training_args.load_best_model_at_end = True
    
    return training_args

# Smaller dataset function for testing
def get_sample_dataset(csv_path, text_column, label_column, sample_size=100):
    """Get a smaller sample of the dataset for testing"""
    try:
        # Load dataset
        df = pd.read_csv(csv_path)
        print(f"Dataset loaded with shape: {df.shape}")
        
        # Check for NaN values
        nan_in_text = df[text_column].isna().sum()
        nan_in_label = df[label_column].isna().sum()
        if nan_in_text > 0 or nan_in_label > 0:
            print(f"Found NaN values - Text column: {nan_in_text}, Label column: {nan_in_label}")
            print("Dropping rows with NaN values...")
            df = df.dropna(subset=[text_column, label_column])
            print(f"Dataset shape after dropping NaN: {df.shape}")
        
        # Get a sample if dataset is larger than sample_size
        if len(df) > sample_size:
            try:
                # Get a stratified sample (handle potential errors)
                label_values = df[label_column].fillna('UNKNOWN').values  # Replace NaNs temporarily for stratification
                df_sample, _ = train_test_split(
                    df, 
                    train_size=sample_size,
                    stratify=label_values if len(set(label_values)) < sample_size else None,
                    random_state=42
                )
                print(f"Using {len(df_sample)} samples out of {len(df)} for testing")
                return df_sample
            except Exception as stratify_error:
                # Fallback to random sampling if stratification fails
                print(f"Stratified sampling failed: {str(stratify_error)}")
                print("Falling back to random sampling...")
                return df.sample(sample_size, random_state=42)
        
        # Return entire dataset if it's smaller than requested sample size
        print(f"Using entire dataset with {len(df)} samples")
        return df
        
    except Exception as e:
        print(f"Error loading sample dataset: {str(e)}")
        # Provide more detailed error information
        import traceback
        traceback.print_exc()
        return None
        
if __name__ == "__main__":
    # Configure settings for faster debugging
    DEBUG_MODE = True  # Set to False for full training
    DATASET_PATH = "/kaggle/input/arabic-mental-health/final_dataset.csv"
    TEXT_COLUMN = "tweet"
    LABEL_COLUMN = "category"
    MAX_LENGTH = 64  # Reduced sequence length for faster processing
    
    print(f"\nStarting with DEBUG_MODE={DEBUG_MODE}")
    print(f"Dataset: {DATASET_PATH}")
    print(f"Text column: {TEXT_COLUMN}, Label column: {LABEL_COLUMN}")
    print(f"Max sequence length: {MAX_LENGTH}")
    
    try:
        if DEBUG_MODE:
            # Use a smaller model for faster debugging
            model_name = "asafaya/bert-base-arabic"  # Smaller model than AraBERT
            num_epochs = 1
            batch_size = 8
            max_length = MAX_LENGTH
            
            # Get a small sample of the dataset
            print("\nLoading sample dataset for debugging...")
            sample_df = get_sample_dataset(DATASET_PATH, TEXT_COLUMN, LABEL_COLUMN, sample_size=100)
            
            if sample_df is not None:
                # Process sample dataset
                sample_df[TEXT_COLUMN] = sample_df[TEXT_COLUMN].apply(preprocess_arabic_text)
                
                # Encode labels
                label_encoder = LabelEncoder()
                sample_df['encoded_labels'] = label_encoder.fit_transform(sample_df[LABEL_COLUMN])
                
                # Split dataset
                train_df, test_df = train_test_split(
                    sample_df, test_size=0.2, random_state=42,
                    stratify=sample_df['encoded_labels']
                )
                
                # Setup model and tokenizer
                print(f"\nSetting up {model_name} model for debugging...")
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_name,
                    num_labels=len(label_encoder.classes_),
                    ignore_mismatched_sizes=True
                ).to(device)
                
                # Create datasets
                train_dataset = ArabicMentalHealthDataset(
                    train_df[TEXT_COLUMN].tolist(),
                    train_df['encoded_labels'].tolist(),
                    tokenizer, max_length=max_length
                )
                
                test_dataset = ArabicMentalHealthDataset(
                    test_df[TEXT_COLUMN].tolist(),
                    test_df['encoded_labels'].tolist(),
                    tokenizer, max_length=max_length
                )
                
                # Get training arguments
                training_args = get_training_args(
                    output_dir="./debug_results",
                    num_epochs=num_epochs,
                    batch_size=batch_size
                )
                
                # Create trainer
                trainer = Trainer(
                    model=model,
                    args=training_args,
                    train_dataset=train_dataset,
                    eval_dataset=test_dataset,
                    compute_metrics=compute_metrics
                )
                
                # Train model
                print("\nStarting debug training...")
                trainer.train()
                
                # Evaluate
                print("\nEvaluating debug model...")
                eval_results = trainer.evaluate()
                print(f"Evaluation results: {eval_results}")
                
                # Test with sample texts
                sample_texts = [
                    "أشعر بالحزن",  # I feel sad
                    "أنا قلق",  # I am anxious
                ]
                
                print("\nTesting debug model with sample texts:")
                for text in sample_texts:
                    predicted_label, confidence = predict_text(
                        model, tokenizer, label_encoder, text
                    )
                    print(f"Text: {text}")
                    print(f"Predicted: {predicted_label} (confidence: {confidence:.3f})")
            
        else:
            # Full training
            print("\nStarting full training pipeline...")
            model, tokenizer, label_encoder = train_and_evaluate(
                csv_path=DATASET_PATH,
                text_column=TEXT_COLUMN,
                label_column=LABEL_COLUMN,
                max_length=MAX_LENGTH
            )
            
            # Test with sample texts
            sample_texts = [
                "أشعر بالحزن والاكتئاب كل يوم",
                "أنا قلق جداً بشأن المستقبل",
                "لا أستطيع النوم بشكل جيد",
                "أحس بالوحدة والضياع"
            ]
            
            print("\nTesting model with sample texts:")
            for text in sample_texts:
                predicted_label, confidence = predict_text(
                    model, tokenizer, label_encoder, text
                )
                print(f"\nText: {text}")
                print(f"Predicted: {predicted_label} (confidence: {confidence:.3f})")
            
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        import traceback
        traceback.print_exc()
