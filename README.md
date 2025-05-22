# P86 Hybrid Multi-Modal Deep Learning for ECG-Based Arrhythmia Classification

A deep learning project focused on multi-modal, hybrid neural network architectures for classifying 12-lead ECG (Electrocardiogram) signals into multiple arrhythmia rhythm categories. The system fuses raw ECG waveform data with patient-specific clinical metadata to improve diagnostic accuracy, leveraging deep residual CNNs, bi-directional GRUs, and a tabular feature MLP for enhanced, context-aware rhythm classification.

---

## Project Structure

```
— data/
│   — Diagnostics.csv            # Diagnostic Labels and Patient Metadata
│   — ECGDataDenoisedDownloaded/ # 12-lead ECG Signals (Downloaded and Denoised)
│   — ECGData/                   # 12-lead Raw ECG Signals (Original) 
│   — ECGDataDenoised/           # 12-lead ECG Signals After Denoising Filters 
│   — RhythmNames.csv/           # Mapping of Rhythm Acronyms To Full Names
│   — AttributesDictionary.csv/  # Description of Metadata Fields in Diagnostics.csv  
— googleColabNotebooks/
│   — ECGDataPreprocessingExploratory.ipynb      # Data Preprocessing and Exploratory Analysis Notebook
│   — ProposedECG-ClassificationExperiment.ipynb # Deep Learning Model Training and Evaluation Notebook
— MyDrive/
│   — bad_files/                  # Corrupted r Invalid ECG Signal Files
│   — ECGExploration2/            # Outputs From Exploratory Data Analysis (EDA)
│   — ECGFiltersOutput/           # Outputs of ECG Denoising and Filtering Experiments
│   — pca_plots_ECG/              # PCA Plots for ECG Data Visualisations (Signals)
│   — results/                    # Results and Plots for the Proposed Hybrid Model
│   — resultsPureCNN/             # Results and Plots for the Pure CNN Model
— README.md                       # Project Documentation
```

---

## Getting Started

### 1. Clone the Repository
If you want to explore or download the project locally:
```bash
git clone https://gitlab.eeecs.qub.ac.uk/your-username/p86-deep-learning-ecg-analysis.git
cd p86-deep-learning-ecg-analysis
```

### 2. Open in Google Colab
This project is primarily designed to be run inside **Google Colab** for access to GPU resources and interactive development.

- Navigate to the `googleColabNotebooks/` directory.
- Open the notebooks:
  - `ECGDataPreprocessingExploratory.ipynb` for data exploration and preprocessing.
  - `ProposedECG-ClassificationExperiment.ipynb` for model training, evaluation, and visualisation.

You can upload the notebooks manually to Google Drive.

### 3. Prepare Data
Ensure the following datasets are available in Google Drive:
- `ECGDataDenoisedDownloaded/`
- `Diagnostics.csv`
- `RhythmNames.csv`
- and other relevant files.

Mount Google Drive inside the Colab notebook by running:
```python
from google.colab import drive
drive.mount('/content/drive')
```

---

> **Note:** 
> Make sure that paths inside the notebooks (e.g., where the data is loaded from) match Google Drive structure after mounting.

---

## Model Architectures

- **CNN Backbone**
  - Multi-stage 1D convolutions with residual connections
  - Squeeze-and-Excitation (SE) blocks for channel attention
  - Spatial attention between stages

- **Bi-Directional GRU**
  - Captures temporal dependencies across ECG waveform sequences
  - Two-layer GRU followed by temporal pooling

- **Auxiliary MLP (Metadata Branch)**
  - Encodes clinical features (age, gender, ventricular rate, etc.)
  - Late fusion with CNN+GRU embeddings

---

## Configuration Options

- **Batch Size:** Settable in `ProposedECG-ClassificationExperiment`
- **Learning Rate:** Tunable in `ProposedECG-ClassificationExperiment`
- **Epochs:** Defined in `ProposedECG-ClassificationExperiment`
- **Device:** Automatically selects CUDA/MPS if available

---

## Outputs

- Training and validation loss/accuracy curves
- Confusion matrices
- ROC-AUC curves
- Saved best model checkpoints (`MyDrive/results`)

---

## Key Features

- **Weighted Random Sampling** to combat class imbalance
- **Adaptive Pooling** for flexible feature extraction
- **Residual Blocks + Attention Mechanisms** for better feature learning
- **Late Fusion** of waveform and metadata information
- **Early Stopping & Learning Rate Scheduler** for robust training

---
