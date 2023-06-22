"""
    script for model training
"""

from __future__ import print_function
import argparse
import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

print("Begging training step")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Hyperparameters are described here. In this simple example we are just including one hyperparameter.
    parser.add_argument("--max_depth", type=int, default=2)

    # Sagemaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument("--output-data-dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--train", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--validation", type=str, default=os.environ["SM_CHANNEL_VALIDATION"])
    args = parser.parse_args()

    # Logging
    print("Arguments:",args)
    print("Dir listing for ", args.train,": ",os.listdir(args.train))

    # Take the set of files and read them all into a single pandas dataframe
    train_input_files = [os.path.join(args.train, file) for file in os.listdir(args.train)]
    if len(train_input_files) == 0:
        raise ValueError(
            (
                "There are no files in {}.\n"
                + "This usually indicates that the channel ({}) was incorrectly specified,\n"
                + "the data specification in S3 was incorrectly specified or the role specified\n"
                + "does not have permission to access the data."
            ).format(args.train, "train")
        )
    train_raw_data = [pd.read_csv(file, header=None, engine="python") for file in train_input_files]
    train_data = pd.concat(train_raw_data)

    # labels are in the first column
    train_y = train_data.iloc[:, 0]
    train_X = train_data.iloc[:, 1:]

    # Take the set of files and read them all into a single pandas dataframe
    valid_input_files = [os.path.join(args.validation, file) for file in os.listdir(args.validation)]
    if len(valid_input_files) == 0:
        raise ValueError(
            (
                "There are no files in {}.\n"
                + "This usually indicates that the channel ({}) was incorrectly specified,\n"
                + "the data specification in S3 was incorrectly specified or the role specified\n"
                + "does not have permission to access the data."
            ).format(args.valid, "valid")
        )
    valid_raw_data = [pd.read_csv(file, header=None, engine="python") for file in valid_input_files]
    valid_data = pd.concat(valid_raw_data)

    # labels are in the first column
    valid_y = valid_data.iloc[:, 0]
    valid_X = valid_data.iloc[:, 1:]

    # Here we support a single hyperparameter, 'max_depth'. Note that you can add as many
    # as your training my require in the ArgumentParser above.
    max_depth = args.max_depth

    # Now use scikit-learn's decision tree classifier to train the model.
    regr = RandomForestRegressor(max_depth=max_depth, random_state=0)
    regr = regr.fit(train_X, train_y)

    preds = regr.predict(valid_X)

    print("MSE = {}" .format(mean_squared_error(valid_y, preds)))

    # Print the coefficients of the trained classifier, and save the coefficients
    joblib.dump(regr, os.path.join(args.model_dir, "model.joblib"))

def model_fn(model_dir):
    """Deserialized and return fitted model
    Note that this should have the same name as the serialized model in the main method
    """
    regr = joblib.load(os.path.join(model_dir, "model.joblib"))
    return regr

print("Completed training step")