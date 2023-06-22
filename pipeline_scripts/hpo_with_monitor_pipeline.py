import json
import boto3
import sagemaker
import pandas as pd
import pipeline_helper as ph
from sagemaker.network import NetworkConfig
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.workflow.check_job_config import CheckJobConfig
from sagemaker.workflow.execution_variables import ExecutionVariables
from sagemaker.workflow.functions import Join
from sagemaker.model_monitor import DatasetFormat, model_monitoring
from sagemaker.workflow.execution_variables import ExecutionVariables
from sagemaker.inputs import TrainingInput
from sagemaker.workflow.steps import TuningStep
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.processing import ScriptProcessor
from sagemaker.model_metrics import MetricsSource, ModelMetrics, FileSource
from sagemaker.drift_check_baselines import DriftCheckBaselines
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.conditions import ConditionLessThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.functions import JsonGet
from sagemaker.workflow.quality_check_step import (
    DataQualityCheckConfig,
    QualityCheckStep,
)
from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterString,
    ParameterFloat,
    ParameterBoolean
)
from sagemaker.tuner import (
    IntegerParameter,
    CategoricalParameter,
    ContinuousParameter,
    HyperparameterTuner,
)

region = boto3.Session().region_name
sagemaker_session = sagemaker.session.Session()
s3_client = boto3.client("s3")

## Variables
# Common variables
accuracy_mse_threshold = float(ph.get_variable('accuracy_mse_threshold')) # the MSE threshold for accepting a trained model
bucket = ph.get_variable('bucket_name')   # s3 bucket used by pipeline
user_id = ph.get_variable('user_id')
kms_key = ph.get_variable('kms_key')
model_package_group_name = ph.get_variable('model_package_group_name') 
prefix = ph.get_variable('bucket_prefix') # the reserved prefix for this project
recreate_pipelines = ph.get_variable('recreate_pipelines') 

role_arn = ph.get_variable('role_arn')
sg_id = ph.get_variable('sg_id')          # Security Group Id 
subnet_id = ph.get_variable('subnet_id')  # Subnet Id from your VPC

# Pipeline specific variables
pipeline_name = ph.get_variable('pipeline_trainwhpo') # name you chose for your training w/ HPO pipeline
training_data_src = ph.get_variable('bucket_train_prefix')

# Built vars
input_data_uri = f"s3://{bucket}/{prefix}/{training_data_src}/training-data.csv" # where the raw input data is stored
role = role_arn 
security_group_ids = [sg_id]
subnets = [subnet_id]
tags = [{"Key":"user_id","Value":user_id}]

# Print out variables
globvars = globals().copy()
ph.printvars(globvars)

## Pipeline Parameters
framework_version = ph.get_parameter("framework_version","0.23-1")
acc_mse_threshold = ParameterFloat(name="AccuracyMseThreshold", default_value=float(ph.get_parameter("AccuracyMseThreshold",accuracy_mse_threshold)))
monitoring_instance_count = ParameterInteger(name="MonitoringInstanceCount", default_value=ph.get_parameter("MonitoringInstanceCount",1))
monitoring_instance_type = ParameterString(name="MonitoringInstanceType", default_value=ph.get_parameter("MonitoringInstanceType","ml.m5.xlarge"))
processing_instance_count = ParameterInteger(name="ProcessingInstanceCount", default_value=ph.get_parameter("ProcessingInstanceCount",1))
processing_instance_type = ParameterString(name="ProcessingInstanceType", default_value=ph.get_parameter("ProcessingInstanceType","ml.m5.xlarge"))
training_instance_count = ParameterInteger(name="TrainingInstanceCount", default_value=ph.get_parameter("TrainingInstanceCount",1))
training_instance_type = ParameterString(name="TrainingInstanceType", default_value=ph.get_parameter("TrainingInstanceType","ml.m5.xlarge"))
transform_instance_type = ParameterString(name="TransformInstanceType", default_value=ph.get_parameter("TransformInstanceType","ml.m5.xlarge"))

# Data quality check step parameters
register_new_baseline_data_quality = ParameterBoolean(name="RegisterNewDataQualityBaseline", default_value=ph.get_parameter("RegisterNewDataQualityBaseline",True))
skip_check_data_quality = ParameterBoolean(name="SkipDataQualityCheck", default_value=ph.get_parameter("SkipDataQualityCheck",True))

# Static parameters
user_id = ParameterString(name="EmpId", default_value=user_id)
input_data = ParameterString(name="InputData", default_value=input_data_uri)
model_approval_status = ParameterString(name="ModelApprovalStatus", default_value="PendingManualApproval")

# Data quality check step parameters
supplied_baseline_constraints_data_quality = ParameterString(name="DataQualitySuppliedConstraints", default_value="")
supplied_baseline_statistics_data_quality = ParameterString(name="DataQualitySuppliedStatistics", default_value="")

## Uploading scripts to s3
# Uploading pre-processing script to S3
response = s3_client.upload_file('./scripts/training_preprocessing.py',
                                 bucket,
                                 '{}/scripts/training_preprocessing.py'.format(prefix))
# Uploading training script to S3
response = s3_client.upload_file('./scripts/training.py',
                                 bucket,
                                 '{}/scripts/training.py'.format(prefix))
# Uploading evaluation script to S3
response = s3_client.upload_file('./scripts/evaluation.py',
                                 bucket,
                                 '{}/scripts/evaluation.py'.format(prefix))

## Create network config for the processing step
network_config = NetworkConfig(
    security_group_ids=security_group_ids,
    subnets=subnets,
    enable_network_isolation=True,
    encrypt_inter_container_traffic=True,
)

## Pre Processing step
sklearn_processor = SKLearnProcessor(
    framework_version=framework_version,
    instance_type=processing_instance_type,
    instance_count=processing_instance_count,
    base_job_name="sklearn-training-preprocess",
    role=role,
    output_kms_key = kms_key,
    volume_kms_key = kms_key,
    network_config = network_config,
    tags = tags,
)

step_process = ProcessingStep(
    name="TrainingPreProcessStep",
    processor=sklearn_processor,
    inputs=[
        ProcessingInput(source=input_data, destination="/opt/ml/processing/input"),
    ],
    outputs=[
        ProcessingOutput(output_name="train", source="/opt/ml/processing/train", destination=f"s3://{bucket}/{prefix}/training/data/processed/train"),
        ProcessingOutput(output_name="validation", source="/opt/ml/processing/validation", destination=f"s3://{bucket}/{prefix}/training/data/processed/validation"),
        ProcessingOutput(output_name="test", source="/opt/ml/processing/test", destination=f"s3://{bucket}/{prefix}/training/data/processed/test"),
    ],
    code=f"s3://{bucket}/{prefix}/scripts/training_preprocessing.py",
)

# Data Quality Check Step
check_job_config = CheckJobConfig(
    role=role,
    instance_count=monitoring_instance_count,
    instance_type=monitoring_instance_type,
    volume_size_in_gb=30,
    sagemaker_session=sagemaker_session,
    volume_kms_key=kms_key,
    output_kms_key=kms_key,
    network_config = network_config,
    tags = tags,
)

data_quality_check_config = DataQualityCheckConfig(
    baseline_dataset=step_process.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
    dataset_format=DatasetFormat.csv(header=False, output_columns_position="START"),
    output_s3_uri=Join(
        on="/",
        values=[
            "s3:/",
            bucket,
            prefix,
            "monitoring",
            ExecutionVariables.PIPELINE_EXECUTION_ID,
            "dataqualitycheckstep",
        ],
    ),
)

data_quality_check_step = QualityCheckStep(
    name="DataQualityCheckStep",
    skip_check=skip_check_data_quality,
    register_new_baseline=register_new_baseline_data_quality,
    quality_check_config=data_quality_check_config,
    check_job_config=check_job_config,
    supplied_baseline_statistics=supplied_baseline_statistics_data_quality,
    supplied_baseline_constraints=supplied_baseline_constraints_data_quality,
    model_package_group_name=model_package_group_name,
)

# configure parameters for SKLearn Estimator
training_code_location = f's3://{bucket}/{prefix}/training/jobs/source' # where the custom training script will be stored as tar.gz file on S3

# training_base_job_name = 'sklearn-rf' # training job base-name
training_output_path = f's3://{bucket}/{prefix}/training/jobs/output' # where the outputs of training job including model artifacts will be stored

# SageMaker SKLearn image
image_uri = sagemaker.image_uris.retrieve(
    framework="sklearn",
    region=region,
    version=framework_version,
    py_version="py3",
    instance_type=training_instance_type,
)

script_path = "./scripts/training.py"

sklearn_train = SKLearn(
    entry_point=script_path,
    image_uri = image_uri,
    instance_type=training_instance_type,
    role=role,
    sagemaker_session=sagemaker_session,
    hyperparameters={"max_depth": 2},
    output_kms_key = kms_key,
    volume_kms_key = kms_key,
    network_config = network_config,
    tags = tags,
    code_location = training_code_location,
    output_path = training_output_path,
    content_type = "text/csv",
    data_type = "S3Prefix",
    split_type = "Line"
)

hyperparameter_ranges = {
    "max_depth": IntegerParameter(2, 5),
}

objective_metric_name = "MSE"
objective_type = "Minimize"
metric_definitions = [{"Name": "MSE", "Regex": "MSE = ([0-9\\.]+)"}]

# Hyperparameter Tuning Step
tuner = HyperparameterTuner(
    sklearn_train,
    objective_metric_name,
    hyperparameter_ranges,
    metric_definitions,
    max_jobs=2,
    max_parallel_jobs=1,
    objective_type=objective_type,
)

step_tuning = TuningStep(
    name = "HPTuning",
    tuner =tuner,
    inputs = {
        "train": TrainingInput(
            s3_data=step_process.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
            content_type="text/csv",
        ),
        "validation": TrainingInput(
            s3_data=step_process.properties.ProcessingOutputConfig.Outputs["validation"].S3Output.S3Uri,
            content_type="text/csv",
        ),
    },
)

# Evaluation Step
sklearn_eval = ScriptProcessor(
    image_uri=image_uri,
    command=["python3"],
    instance_type=processing_instance_type,
    instance_count=processing_instance_count,
    base_job_name="script-eval",
    role=role,
    network_config = network_config,
    tags = tags,
)

model_bucket_key = f'{bucket}/{prefix}/training/jobs/output'

evaluation_report = PropertyFile(
    name="EvaluationReport", output_name="evaluation", path="evaluation.json"
)

step_eval = ProcessingStep(

    name="TrainingEval",
    processor=sklearn_eval,
    inputs=[
        ProcessingInput(
            source=step_tuning.get_top_model_s3_uri(top_k=0, s3_bucket=model_bucket_key),
            destination="/opt/ml/processing/model",
        ),
        ProcessingInput(
            source=step_process.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,
            destination="/opt/ml/processing/test",
        ),
    ],
    outputs=[
        ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation",
                         destination=Join(
                                        on="/",
                                        values=[
                                            "s3://{}".format(bucket),
                                            prefix,
                                            "training/evaluation",
                                            ExecutionVariables.PIPELINE_EXECUTION_ID,
                                            "evaluation-report",
                                        ],
                                    )
                        )
    ],
    code=f"s3://{bucket}/{prefix}/scripts/evaluation.py",
    property_files=[evaluation_report],
)

model_metrics = ModelMetrics(
    model_data_statistics=MetricsSource(
        s3_uri=data_quality_check_step.properties.CalculatedBaselineStatistics,
        content_type="application/json",
    ),
    model_data_constraints=MetricsSource(
        s3_uri=data_quality_check_step.properties.CalculatedBaselineConstraints,
        content_type="application/json",
    ),
    model_statistics=MetricsSource(
        s3_uri = Join(
                                        on="/",
                                        values=[
                                            "s3://{}".format(bucket),
                                            prefix,
                                            "training/evaluation",
                                            ExecutionVariables.PIPELINE_EXECUTION_ID,
                                            "evaluation-report",
                                            "evaluation.json"
                                        ],
                                    ),
        content_type="application/json",
    )
)

drift_check_baselines = DriftCheckBaselines(
    model_data_statistics=MetricsSource(
        s3_uri=data_quality_check_step.properties.BaselineUsedForDriftCheckStatistics,
        content_type="application/json",
    ),
    model_data_constraints=MetricsSource(
        s3_uri=data_quality_check_step.properties.BaselineUsedForDriftCheckConstraints,
        content_type="application/json",
    ),
)

# Registering Model
step_register = RegisterModel(
    name="ModelRegisterStep",
    estimator=sklearn_train,
    model_data=step_tuning.get_top_model_s3_uri(top_k=0, s3_bucket=model_bucket_key),
    content_types=["text/csv"],
    response_types=["text/csv"],
    inference_instances=["ml.t2.medium", "ml.m5.xlarge"],
    transform_instances=[transform_instance_type],
    model_package_group_name=model_package_group_name,
    approval_status=model_approval_status,
    model_metrics=model_metrics,
    tags = tags
)

cond_lte = ConditionLessThanOrEqualTo(
    left=JsonGet(
        step_name=step_eval.name,
        property_file=evaluation_report,
        json_path="regression_metrics.mse.value",
    ),
    right=acc_mse_threshold,
)

step_cond = ConditionStep(
    name="MSECond",
    conditions=[cond_lte],
        if_steps=[step_register],
    else_steps=[],
)

# Create the pipeline object
pipeline = Pipeline(
    name=pipeline_name,
    parameters=[
        acc_mse_threshold,
        user_id,
        input_data,
        model_approval_status,
        monitoring_instance_count,
        monitoring_instance_type,
        processing_instance_count,
        processing_instance_type,
        register_new_baseline_data_quality,
        skip_check_data_quality,
        supplied_baseline_constraints_data_quality,
        supplied_baseline_statistics_data_quality,
        training_instance_count,
        training_instance_type,
        transform_instance_type,
    ],
    steps=[step_process, data_quality_check_step, step_tuning, step_eval, step_cond],
)

print(pipeline)

# Validate the pipeline definition
definition = json.loads(pipeline.definition())
print(definition)

# Update/create the pipeline
# If set, delete existing pipeline first
if recreate_pipelines == "true":
    try:
        sm_client = boto3.client("sagemaker")
        sm_client.delete_pipeline(PipelineName=pipeline_name)
        print('Pipeline {} has been deleted'.format(pipeline_name))
    except Exception as e:
        print('An error occurred: {} '.format(e))

pipeline.upsert(role_arn=role)

print('Pipeline {} has been created'.format(pipeline_name))