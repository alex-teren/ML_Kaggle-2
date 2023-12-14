# Toxic Comment Classification Competition

## Overview
This repository contains the code and models for the Toxic Comment Classification Competition on [Kaggle](https://www.kaggle.com/competitions/kmaml223/). The goal is to build a model that can classify comments into various toxicity categories.

## Repository Structure
- `.github/workflows`: Contains GitHub Actions for CI/CD.
- `EDA`: Jupyter notebooks for exploratory data analysis.
- `configs`: Configuration files for model training.
- `src`: Source code for training and inference.
- `tests`: Test cases for the source code.
- `requirements.txt`: List of Python package dependencies.

## Getting Started
To set up the project, run `pip install -r requirements.txt` to install the required dependencies.

## Usage
To train the model, run `python src/train.py` with the appropriate configuration file.
To perform inference, run `python -m src.inference "your text here"`.

## Testing
Run `pytest` in the root directory to execute the test suite. The code has a test coverage of at least 40%.
