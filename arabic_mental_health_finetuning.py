# Arabic Mental Health Classification - Fine-tuning Pipeline
# Optimized for Kaggle GPU environment

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
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
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize
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
    """Enhanced Arabic text preprocessing"""
    try:
        # Convert to string if it's not already
        if pd.isna(text):
            return ''
        text = str(text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove mentions and hashtags
        text = re.sub(r'@[\w\u0621-\u064A]+', '', text)
        text = re.sub(r'#\w+', '', text)
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        # Remove emojis and special characters
        text = re.sub(r'[\U0001F600-\U0001F64F]', '', text)  # Emojis
        text = re.sub(r'[\U0001F300-\U0001F5FF]', '', text)  # Symbols
        text = re.sub(r'[\U0001F680-\U0001F6FF]', '', text)  # Transport and map symbols
        text = re.sub(r'[\U0001F1E0-\U0001F1FF]', '', text)  # Flags
        # Remove punctuation
        text = re.sub(r'[\u061B\u061F\u066A-\u066D\u06D4]', '', text)  # Arabic punctuation
        # Normalize Arabic text
        text = re.sub(r'[إأآ]', 'ا', text)  # Normalize different forms of 'a'
        text = re.sub(r'[ؤئ]', 'ي', text)  # Normalize different forms of 'y'
        text = re.sub(r'[ًٌٍَُِّْ]', '', text)  # Remove tashkeel (diacritical marks)
        
        return text.strip()
    except Exception as e:
        print(f"Error processing text: {str(e)}")
        return ''

def load_and_prepare_data(csv_path, text_column, label_column):
    """Load and prepare the dataset with enhanced handling"""
    print("\nLoading and preparing dataset...")
    try:
        df = pd.read_csv(csv_path)
        
        # Basic info
        print(f"\nDataset shape: {df.shape}")
        print(f"Label distribution:\n{df[label_column].value_counts()}\n")
        
        # Preprocess text
        print("Preprocessing Arabic text...")
        df[text_column] = df[text_column].apply(preprocess_arabic_text)
        
        # Remove empty texts after preprocessing
        df = df[df[text_column].str.len() > 0]
        print(f"\nDataset shape after preprocessing: {df.shape}")
        
        # Encode labels
        label_encoder = LabelEncoder()
        df['encoded_labels'] = label_encoder.fit_transform(df[label_column])
        
        # Print label mapping
        label_mapping = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))
        print("\nLabel Mapping:")
        for label, idx in label_mapping.items():
            print(f"{idx}: {label}")
        
        return df, label_encoder
    except Exception as e:
        print(f"Error preparing dataset: {str(e)}")
        raise

def setup_model_and_tokenizer(model_name="aubmindlab/bert-base-arabertv2", num_labels=None):
    """Initialize model and tokenizer"""
    print(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        ignore_mismatched_sizes=True
    )
    
    return model, tokenizer

def compute_metrics(eval_pred):
    """Compute metrics for evaluation"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    return {"accuracy": accuracy}

def create_trainer(model, tokenizer, train_dataset, val_dataset, output_dir="./results"):
    """Create and configure trainer"""
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,  # Start with 3 epochs for Kaggle time limits
        per_device_train_batch_size=16,  # Adjust based on GPU memory
        per_device_eval_batch_size=32,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=100,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        report_to=None,  # Disable wandb for Kaggle
        fp16=True,  # Enable mixed precision for faster training
        dataloader_num_workers=2,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    return trainer

def evaluate_model(trainer, test_dataset, label_encoder):
    """Evaluate the model and show detailed results"""
    print("Evaluating model...")
    
    # Get predictions
    predictions = trainer.predict(test_dataset)
    y_pred = np.argmax(predictions.predictions, axis=1)
    y_true = predictions.label_ids
    
    # Convert back to original labels
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
    plt.show()
    
    return y_true_labels, y_pred_labels

def main_pipeline(csv_path, text_column, label_column):
    """Main training pipeline"""
    
    # 1. Load and prepare data
    df, label_encoder = load_and_prepare_data(csv_path, text_column, label_column)
    
    # 2. Split data
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        df[text_column].tolist(), 
        df['encoded_labels'].tolist(),
        test_size=0.3, 
        random_state=42, 
        stratify=df['encoded_labels']
    )
    
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels,
        test_size=0.5, 
        random_state=42, 
        stratify=temp_labels
    )
    
    print(f"Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")
    
    # 3. Setup model and tokenizer
    num_labels = len(label_encoder.classes_)
    model, tokenizer = setup_model_and_tokenizer(
        model_name="aubmindlab/bert-base-arabertv2",  # Great Arabic BERT model
        num_labels=num_labels
    )
    
    # 4. Create datasets
    train_dataset = ArabicMentalHealthDataset(train_texts, train_labels, tokenizer)
    val_dataset = ArabicMentalHealthDataset(val_texts, val_labels, tokenizer)
    test_dataset = ArabicMentalHealthDataset(test_texts, test_labels, tokenizer)
    
    # 5. Create trainer and train
    trainer = create_trainer(model, tokenizer, train_dataset, val_dataset)
    
    print("Starting training...")
    trainer.train()
    
    # 6. Evaluate
    y_true, y_pred = evaluate_model(trainer, test_dataset, label_encoder)
    
    # 7. Save model
    model.save_pretrained("./arabic-mental-health-model")
    tokenizer.save_pretrained("./arabic-mental-health-model")
    print("Model saved to ./arabic-mental-health-model")
    
    return trainer, model, tokenizer, label_encoder

# Example usage and testing function
def test_model(model, tokenizer, label_encoder, sample_texts):
    """Test the model with sample texts"""
    model.eval()
    
    print("\nTesting model with sample texts:")
    for text in sample_texts:
        # Preprocess
        clean_text = preprocess_arabic_text(text)
        
        # Tokenize
        inputs = tokenizer(
            clean_text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )
        
        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(predictions, dim=-1)
            confidence = torch.max(predictions).item()
        
        # Convert to label
        predicted_label = label_encoder.inverse_transform([predicted_class.item()])[0]
        
        print(f"Text: {text[:100]}...")
        print(f"Predicted: {predicted_label} (confidence: {confidence:.3f})")
        print("-" * 50)

# Dataset verification
def verify_dataset(csv_path, text_column, label_column):
    """Verify dataset structure and content"""
    try:
        df = pd.read_csv(csv_path)
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in dataset")
        if label_column not in df.columns:
            raise ValueError(f"Column '{label_column}' not found in dataset")
        
        # Check for missing values
        missing_text = df[text_column].isna().sum()
        missing_labels = df[label_column].isna().sum()
        
        if missing_text > 0:
            print(f"Warning: Found {missing_text} missing text entries")
        if missing_labels > 0:
            print(f"Warning: Found {missing_labels} missing label entries")
            
        print("\nDataset Verification:")
        print(f"Total entries: {len(df)}")
        print(f"Text column: {text_column} (dtype: {df[text_column].dtype})")
        print(f"Label column: {label_column} (dtype: {df[label_column].dtype})")
        print("\nLabel distribution:")
        print(df[label_column].value_counts())
        
        return True
    except Exception as e:
        print(f"Error verifying dataset: {str(e)}")
        return False

# Kaggle-specific optimization tips
def kaggle_optimization_tips():
    """Tips for running on Kaggle efficiently"""
    print("\nKaggle Optimization Tips:")
    print("1. Using mixed precision (fp16) for faster training")
    print("2. Optimized batch sizes for GPU memory")
    print("3. Early stopping to prevent overfitting")
    print("4. Limited epochs to work within Kaggle's time limits")
    print("5. Proper logging and monitoring setup")
    print("6. Save checkpoints regularly")

if __name__ == "__main__":
    # Show optimization tips
    kaggle_optimization_tips()
    
    # Example usage (uncomment and modify for your dataset):
    # Replace with your actual file path and column names
    CSV_PATH = "/kaggle/input/arabic-mental-health/final_dataset.csv"
    TEXT_COLUMN = "tweet"  # Your text column name
    LABEL_COLUMN = "category"  # Your label column name
    
    # Run the pipeline
    trainer, model, tokenizer, label_encoder = main_pipeline(
        CSV_PATH, TEXT_COLUMN, LABEL_COLUMN
    )
    
    # Test with sample texts
    sample_texts = [
        "أشعر بالحزن والاكتئاب كل يوم",  # I feel sad and depressed every day
        "أنا قلق جداً بشأن المستقبل",      # I'm very anxious about the future
    ]
    test_model(model, tokenizer, label_encoder, sample_texts)
    
    print("Pipeline ready! Update the file paths and column names, then run main_pipeline()")
