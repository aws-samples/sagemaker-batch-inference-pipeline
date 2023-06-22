"""
    script to monitor trained model and identify drifts from baseline.
"""

import os
import json
import re
import logging

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logger = logging.getLogger()
logger.setLevel(LOG_LEVEL)
region = os.environ.get('Region', 'NoAWSRegionFound')
pipeline_name = os.environ.get('PipelineName', 'NoPipelineNameFound')

def get_baseline_drift(feature):
    if "violations" in feature:
        for violation in feature["violations"]:
            if violation["constraint_check_type"] == "baseline_drift_check":
                desc = violation["description"]
                matches = re.search("distance: (.+) exceeds threshold: (.+)", desc)
                if matches:
                    yield {
                        "metric_name": f'feature_baseline_drift_{violation["feature_name"]}',
                        "metric_value": float(matches.group(1)),
                        "metric_threshold": float(matches.group(2)),
                    }

def postprocess_handler():
    violations_file = "/opt/ml/processing/output/constraint_violations.json"
    if os.path.isfile(violations_file):
        f = open(violations_file)
        violations = json.load(f)
        metrics = list(get_baseline_drift(violations))
        logger.info("Constraint violations found:")
        logger.debug(metrics)
    else:
        logger.info("No constraint_violations file found. All good!")
