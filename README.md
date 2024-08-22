# LSTM-Based EEG Reaction Time Prediction

## Overview

This repository contains the implementation of an LSTM-based model for predicting reaction times from EEG signals of both healthy and Parkinsonian subjects. The project was conducted as part of a research initiative at the University of British Columbia (UBC).

## Project Details

- **Objective:** To develop and evaluate a deep learning model using Long Short-Term Memory (LSTM) networks for analyzing EEG data and predicting reaction times in different subject groups.
- **Subjects:** The dataset includes EEG signals from both healthy individuals and those diagnosed with Parkinson's disease.
- **Methodology:** The project involves preprocessing EEG data, training an LSTM model, and analyzing the model's performance across the different subject groups.

## Repository Contents

- `data/`: Contains example EEG data files (Note: real data may need to be sourced or requested separately due to privacy).
- `models/`: Contains the LSTM model architecture and weights.
- `notebooks/`: Jupyter notebooks used for data exploration, preprocessing, and model training.
- `scripts/`: Python scripts for running the LSTM model and generating predictions.
- `results/`: Outputs and performance metrics from the model, including reaction time predictions.

## Usage

1. **Preprocessing:**
   - The `notebooks/preprocessing.ipynb` notebook demonstrates how to preprocess EEG data for input into the LSTM model.

2. **Training the Model:**
   - Use the `notebooks/train_model.ipynb` to train the LSTM model on your dataset.

3. **Evaluating Performance:**
   - The `notebooks/evaluate_model.ipynb` notebook provides an analysis of the model's performance across different subject groups.

## Requirements

- Python 3.x
- PyTorch
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

Install the required Python packages using:

```bash
pip install -r requirements.txt
