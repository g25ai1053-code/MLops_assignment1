"""
train2.py
---------
Trains a KernelRidge model on the Boston Housing dataset using the SAME generic
helper functions defined in misc.py, then reports the average MSE on the test
set.
"""

from sklearn.kernel_ridge import KernelRidge

from misc import (
    load_data,
    preprocess_data,
    train_model,
    test_model,
    average_mse_cv,
)


def main():
    # 1. Load the data
    df = load_data()

    # 2. Preprocess: train/test split + scaling
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # 3. Define the model
    model = KernelRidge(alpha=1.0, kernel="rbf")

    # 4. Train
    model = train_model(model, X_train, y_train)

    # 5. Test (single hold-out MSE)
    test_mse = test_model(model, X_test, y_test)

    # 6. Average MSE via cross-validation on the full dataset
    X = df.drop(columns=["MEDV"]).values
    y = df["MEDV"].values
    avg_mse = average_mse_cv(model, X, y, cv=5)

    print("=" * 50)
    print("Model: KernelRidge")
    print("=" * 50)
    print(f"Test set MSE        : {test_mse:.4f}")
    print(f"Average MSE (5-fold): {avg_mse:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
