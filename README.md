# CNN + Metadata Models for Cough Audio Spectrogram Classification

This project implements various deep learning models to classify cough spectrograms as **Healthy**, **Symptomatic**, or **COVID-19** using a combination of image and metadata inputs. It includes both **CNN-based** and **ViT-based** architectures.

## Project Structure

This notebook processes image data alongside metadata from the CoughVID dataset, and trains multiple models for both binary and trinary classification tasks.

## Setup Instructions

### 1. Google Colab Recommended
Run the notebook on Google Colab for best performance with GPU acceleration.

### 2. Install Dependencies
```python
!pip install torch torchvision torchviz transformers
```

## Dataset Preparation

### A. Mount Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

### B. Load the CSV metadata file
Replace this with the path to your actual CSV:
```python
df_original = pd.read_csv("/content/drive/MyDrive/ML_Files/coughvid_v3.csv")
```

### C. Load Spectrogram Images (128x128 or 600x240)
Uncomment and update path accordingly:
```python
dataset = load_images_with_filenames("/path/to/image_folder", target_size=(128, 128))
```

Save or load preprocessed `.npy` image data:
```python
# Save dataset (optional)
save_image_dataset(dataset, "/content/images_dataset.npy")

# Load dataset
dataset = np.load("/content/images_dataset.npy")
```

## Data Processing

### Merge Metadata + Image Arrays
```python
merged_df = merge_image_array_with_df(df_original, dataset)
```

### Clean and Filter Dataset
```python
clean_df = merged_df.dropna(subset=['image', 'status'])
```

### Preprocessing Modes:
- One-hot encode for Trinary classification (`healthy`, `symptomatic`, `COVID-19`)
- Create binary labels for Binary classification (`healthy`, `unhealthy`)

```python
clean_df_binary['status_binary'] = (clean_df_binary['status'] != 'healthy').astype(int)
```

## Model Architectures

### 1. CNN + Metadata (Binary Classification)
File saved as: `best_binary_cnn_with_meta.pt`

### 2. CNN + Metadata (Trinary Classification)
File saved as: `best_trinary_cnn_with_meta.pt`

### 3. Transformer (ViT) + Metadata (Binary)
```python
ViTWithMetaSimplified()
```
Pretrained on `google/vit-base-patch16-224-in21k`

File saved as: `best_vit_with_meta_finetune.pt`

## Training and Evaluation

All training loops include:
- Accuracy tracking
- Validation loss monitoring
- Best model checkpointing
- Confusion matrix and classification report display

#### Example:
```python
print(classification_report(y_true, y_pred, target_names=['healthy', 'unhealthy']))
```

## Switching Between Models

To switch training tasks:
- Binary CNN: Set `clean_df_binary` as your main dataset
- Trinary CNN: Use `balanced_df_ohe`
- Focal Loss versions: Enable FocalLoss class and change `criterion`
- Transformer: Switch to `ViTMetaDataset`, update transforms and model

## Notes

Be Sure to:
- Update all file paths to match your local or Drive directory.
- Uncomment only the sections you need.
- Save your best model using `torch.save(...)`.

## Sample Confusion Matrix

All models produce a confusion matrix like:

```
Predicted →
          healthy | unhealthy
Actual ↓
healthy      90         10
unhealthy    15         85
```

## Future Improvements

- Use spectrogram augmentation (time shift, noise)
- Hyperparameter tuning via Optuna or RayTune
- Larger ViT models with frozen image features

## Authors

- Ethan Soroko (https://github.com/EthanSoroko)
- Theodore Utomo (https://github.com/Theodore-Utomo)

## License

MIT License
