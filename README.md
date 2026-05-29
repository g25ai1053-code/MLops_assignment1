# MLOps Assignment 1 — Boston Housing Price Prediction

This project sets up an end-to-end machine learning pipeline to estimate house prices using the **Boston Housing** dataset, with multiple scikit-learn regression models.

## Project Layout

| Branch        | Important Files                          | Algorithm               |
|---------------|------------------------------------------|-------------------------|
| `main`        | `README.md`                              | —                       |
| `dtree`       | `train.py`, `misc.py`, `requirements.txt`| DecisionTreeRegressor   |
| `kernelridge` | `train2.py`, `misc.py`, CI workflow      | KernelRidge             |

All preprocessing, training, and evaluation logic is written inside `misc.py` as reusable functions, making it easy to plug in any scikit-learn regressor.

## Setup Instructions

1. Create and activate a conda environment:

   ```bash
   conda create -n mlops python=3.11 -y
   conda activate mlops
   ```

2. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

## How to Run

To train and evaluate the Decision Tree model:

```bash
python train.py
```

To train and evaluate the Kernel Ridge model:

```bash
python train2.py
```

Both scripts will output the test-set MSE along with the average 5-fold cross-validation MSE.

## CI/CD Pipeline

A GitHub Actions workflow (`.github/workflows/ci.yml`) is triggered on every push to the `kernelridge` branch. It automatically checks out the repository, installs all dependencies, and executes both `train.py` and `train2.py`, logging the performance metrics for each model.
