"""
    script for pre-processing of data quality check
"""

import logging
import pathlib
import pandas as pd
import numpy as np

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

print("Beginning data check preprocessing step")

if __name__ == "__main__":
    base_dir = "/opt/ml/processing"
    
    logger.debug("Starting pre-processing.")
    
    logger.debug("Reading batch inference results.")
    with open(f"{base_dir}/inference_results/inference-data-processed.csv.out") as fp:
        results = fp.read()
        
    y = [float(item) for item in results.replace('[','').replace(']','').replace(' ','').split(',')]
    
    Y = pd.DataFrame(y)
    
    logger.debug("Reading input data.")
    X = pd.read_csv(f"{base_dir}/inference_data/inference-data-processed.csv", 
                    header=None)

    logger.debug("Merging batch inference input and output.")
    df = pd.DataFrame(np.concatenate((Y.values,X.values),axis=1))

    output_dir = f"{base_dir}/output"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info("Writing out the merged data")
    df.to_csv(f"{output_dir}/inference-data-combined.csv", index=False, header=False)

print("Finished data check preprocessing step")
