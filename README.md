# Protein Fitness Prediction with Active Learning 

This repository contains a Python implementation for predicting protein fitness scores using embeddings extracted from the ESM model. The pipeline includes data preprocessing, model training, active learning, and evaluation.

## Features
Protein Sequence Mutation Handling: Generate mutated protein sequences based on input mutations.

Embedding Extraction: Use the ESM model to extract embeddings for protein sequences.

Fitness Prediction Model: Train a neural network to predict fitness scores based on sequence embeddings.

Active Learning Loop: Select samples with high uncertainty for further labeling and model improvement.

Evaluation: Assess model performance using Spearman's correlation and generate predictions.

## Dependencies

The following Python libraries are required:


fair-esm: Pretrained ESM models for protein sequence embeddings.

torch: PyTorch framework for deep learning.

pandas: Data manipulation and analysis.

numpy: Numerical computations.

scipy: Statistical computations.

sklearn: Machine learning utilities.

Install dependencies using:

```
pip install fair-esm torch pandas numpy scipy scikit-learn
```

## How to Run

1. You must have required dependendencies installed.

2. Since this is a .ipynb file, you must execute in Jupyter Notebook.If you dont have Jupyter installed, you can install it in terminal using this command: 
   
   ```
   pip install notebook
   ```

3. Once installed, navigate to the folder with the input files and G22_Hackathon.ipynb and run this:
     
   ```
   jupyter notebook
   ```

## Project Structure

### Key Sections 

#### Initial Setup and Data Loading:

Load protein sequences and mutation data from FASTA files and CSV datasets.

Generate mutated sequences based on input mutations.

#### Model Preparation:

Extract embeddings for protein sequences using the ESM model.

Define a neural network architecture for fitness prediction.

#### Initial Training:

Train the fitness predictor model using labeled data.

Implement early stopping to prevent overfitting.

#### Active Learning Loop:

Select samples with high uncertainty for further labeling.

Update the training dataset and retrain the model iteratively.

#### Final Evaluation:

Evaluate the model's predictions against new labeled data.

Compute Spearman's correlation coefficient to assess performance.

## Functions

get_mutated_sequence(mut, sequence_wt): Generates mutated protein sequences based on input mutations.

extract_embeddings(sequences, batch_size=12): Extracts embeddings for protein sequences using the ESM model.

train_model(model, train_loader, val_loader, optimizer, criterion, scheduler, num_epochs=150): Trains the fitness predictor model.

acquisition_function(model, unlabeled_embeddings): Computes uncertainties for active learning.

select_samples(model, unlabeled_pool, n_samples=100): Selects samples based on uncertainties.

generate_query(model, test_set, n_queries=100): Generates query points for active learning.

## Input Files

#### The following datasets are required:

sequence.fasta: Contains the wild-type protein sequence in FASTA format.

train.csv: Training dataset with mutants and corresponding fitness scores (DMS_score). Must contain 2 columns, "mutant" and "DMS_score". 

test.csv: Test dataset with mutations for querying during active learning. Must contain one column, "mutant".

new_dms_data.csv: Newly labeled data generated during active learning, produced by Gradescope. Must contain the columns "mutant", "DMS_score" and "sequence".

## Outputs

The pipeline generates the following files:

query.txt: List of selected mutations for querying during active learning.

predictions.csv: Predicted fitness scores for test mutations.

top10.txt: Top 10 mutants with highest predicted fitness scores.

## Usage

#### Step 1: Initial Training

Run the script to train the fitness predictor model using the training dataset (train.csv).

#### Step 2: Active Learning

Generate queries (query.txt) using the active learning loop. After labeling new data (new_dms_data.csv), retrain the model.

#### Step 3: Final Evaluation

Evaluate predictions on test data (test.csv) and compute Spearman's correlation coefficient.
