"""
    script for evaluating inference results of the trained model with ground truth values.
"""

import json
import pathlib
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

print("Begining inference evaluation step")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--mse-threshold", type=float, default=10.0)
    args, _ = parser.parse_known_args()

    print("Received arguments {}".format(args))
    
    base_dir = "/opt/ml/processing"
    
    with open(f"{base_dir}/inference_results/inference-data-processed.csv.out") as fp:
        results = fp.read()
        
    y = [float(item) for item in results.replace('[','').replace(']','').replace(' ','').split(',')]
    predictions = pd.DataFrame(y)
    
    ground_truth = pd.read_csv(f"{base_dir}/ground_truth/ground-truth.csv", 
                    header=None)

    mse = mean_squared_error(ground_truth.values, predictions.values)
    print("MSE found")
    print(mse)
    std = np.std(ground_truth.values - predictions.values)

    mse_thresh = args.mse_threshold
    model_quality_passed = True if mse<mse_thresh else False

    report_dict = {
        "regression_metrics": {
            "mse": {"value": mse, "standard_deviation": std},
        },
        "model_quality_passed": model_quality_passed,
        "mse_threshold": mse_thresh
    }

    output_dir = "/opt/ml/processing/output"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    evaluation_path = f"{output_dir}/inference-evaluation.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))

    if not model_quality_passed:
        raise Exception("Model performance was below threshold")

    print("Finished inference evaluation step")
