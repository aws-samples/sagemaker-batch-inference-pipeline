import boto3
import json
import sagemaker
import pandas as pd
import pipeline_helper as ph
from sagemaker.network import NetworkConfig
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.model import Model
from sagemaker.inputs import CreateModelInput
from sagemaker.workflow.steps import CreateModelStep
from sagemaker.transformer import Transformer
from sagemaker.workflow.functions import Join
from sagemaker.workflow.execution_variables import ExecutionVariables
from sagemaker.inputs import TransformInput
from sagemaker.workflow.steps import TransformStep
from sagemaker.processing import ScriptProcessor
from sagemaker.workflow.check_job_config import CheckJobConfig
from sagemaker.model_monitor import DatasetFormat, model_monitoring
from sagemaker.workflow.conditions import ConditionEquals
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.quality_check_step import (
    DataQualityCheckConfig,
    QualityCheckStep,
)
from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterString,
    ParameterBoolean,
    ParameterFloat
)

print('Creating pipeline {}'.format(ph.get_variable('pipeline_inf')))

# Boto3 client and session
region = boto3.Session().region_name
sagemaker_session = sagemaker.session.Session()
sm_client = boto3.client("sagemaker") 
s3_client = boto3.client("s3")

## Variables
# Common variables
accuracy_mse_threshold = float(ph.get_variable('accuracy_mse_threshold')) # the MSE threshold for model quality check
bucket = ph.get_variable('bucket_name') # s3 bucket used by pipeline
user_id = ph.get_variable('user_id')
kms_key = ph.get_variable('kms_key')
model_package_group_name = ph.get_variable('model_package_group_name') 
prefix = ph.get_variable('bucket_prefix') # the reserved prefix for this project

role_arn = ph.get_variable('role_arn')
sg_id = ph.get_variable('sg_id') # SECURITY GROUP ID 
subnet_id = ph.get_variable('subnet_id') # SUBNET ID FROM YOUR VPC
recreate_pipelines = ph.get_variable('recreate_pipelines')

# Pipeline specific variables
inference_prefix = ph.get_variable('bucket_inf_prefix')
pipeline_name = ph.get_variable('pipeline_inf') # name you chose for your inference pipeline
upload_inference_data = ph.get_variable('upload_inference_data')

# Built vars
input_data_uri =  f"s3://{bucket}/{prefix}/{inference_prefix}/inference-data.csv" # where the raw input data is stored
role = role_arn  
scoring_results_uri =  f"s3://{bucket}/{prefix}/batch-scoring/results" # where we want inference results to be stored
security_group_ids = [sg_id]
subnets = [subnet_id]
tags = [{"Key":"user_id","Value":user_id}]

# Print out variables
globvars = globals().copy()
ph.printvars(globvars)

# Gets list of approved model packages from the model package group we specified earlier
approved_model_packages = sm_client.list_model_packages(
      ModelApprovalStatus='Approved',
      ModelPackageGroupName=model_package_group_name,
      SortBy='CreationTime',
      SortOrder='Descending'
  )

# Get the latest approved model package
try:
    latest_approved_model_package_arn = approved_model_packages['ModelPackageSummaryList'][0]['ModelPackageArn']
    print("Found latest approved model package:", latest_approved_model_package_arn)
    # retrieve required information about the model
    latest_approved_model_package_descr =  sm_client.describe_model_package(ModelPackageName = latest_approved_model_package_arn)
except Exception as e:
    print("Failed to retrieve an approved model package:", e)

# Retrieve information about the training job from the latest approved model
training_job_name = latest_approved_model_package_descr['InferenceSpecification']['Containers'][0]['ModelDataUrl'].split('/')[-3]
training_job_descr = sm_client.describe_training_job(TrainingJobName = training_job_name)

# Training script directory path (tar.gz file)
sagemaker_submit_directory = training_job_descr['HyperParameters']['sagemaker_submit_directory'][1:-1]

# Custom training script name (in the above compressed file)
sagemaker_program = training_job_descr['HyperParameters']['sagemaker_program'][1:-1]

# Trained Model artifact uri (tar.gz file)
model_artifact_uri = latest_approved_model_package_descr['InferenceSpecification']['Containers'][0]['ModelDataUrl']

# Sagemaker Image URI in ECR
image_uri = latest_approved_model_package_descr['InferenceSpecification']['Containers'][0]['Image']

# Baseline statistics for data quality
data_quality_baseline_statistics = latest_approved_model_package_descr['ModelMetrics']['ModelDataQuality']['Statistics']['S3Uri']

# Baseline constraints for data quality
data_quality_baseline_constraints = latest_approved_model_package_descr['ModelMetrics']['ModelDataQuality']['Constraints']['S3Uri']

## Pipeline Parameters
framework_version = ph.get_parameter("framework_version","0.23-1")

acc_mse_threshold = ParameterString(name="AccuracyMseThreshold", default_value=str(ph.get_parameter("AccuracyMseThreshold", accuracy_mse_threshold)))
processing_instance_type = ParameterString(name="ProcessingInstanceType", default_value=ph.get_parameter("ProcessingInstanceType","ml.m5.xlarge"))
processing_instance_count = ParameterInteger(name="ProcessingInstanceCount", default_value=ph.get_parameter("ProcessingInstanceCount",1))
transform_instance_type = ParameterString(name="TransformInstanceType", default_value=ph.get_parameter("TransformInstanceType","ml.m5.xlarge"))
transform_instance_count = ParameterInteger(name="TransformInstanceCount", default_value=ph.get_parameter("TransformInstanceCount",1))
monitoring_instance_count = ParameterInteger(name="MonitoringInstanceCount", default_value=ph.get_parameter("MonitoringInstanceCount",1))
monitoring_instance_type = ParameterString(name="MonitoringInstanceType", default_value=ph.get_parameter("MonitoringInstanceType","ml.m5.xlarge"))

# Data quality check step parameters
skip_check_data_quality = ParameterBoolean(name="SkipDataQualityCheck", default_value=ph.get_parameter("SkipDataQualityCheck",False))
register_new_baseline_data_quality = ParameterBoolean(name="RegisterNewDataQualityBaseline", default_value=ph.get_parameter("RegisterNewDataQualityBaseline",False))

# Static parameters
input_data_path = ParameterString(name="InputData", default_value=input_data_uri)
scoring_results_path =  ParameterString(name="ScoringResults", default_value=scoring_results_uri)
user_id = ParameterString(name="EmpId", default_value=user_id)

# Data quality check step parameters
supplied_baseline_statistics_data_quality = ParameterString(name="DataQualitySuppliedStatistics", default_value=data_quality_baseline_statistics)
supplied_baseline_constraints_data_quality = ParameterString(name="DataQualitySuppliedConstraints", default_value=data_quality_baseline_constraints)

## Uploading scripts to s3
# Uploading scoring script to S3
response = s3_client.upload_file('./scripts/scoring_preprocessing.py',
                                 bucket,
                                 '{}/scripts/scoring_preprocessing.py'.format(prefix))
# Uploading data check script to S3
response = s3_client.upload_file('./scripts/data_check_preprocessing.py', 
                                 bucket,
                                 '{}/scripts/data_check_preprocessing.py'.format(prefix))
# Uploading post process monitoring script to S3
response = s3_client.upload_file('./scripts/postprocess_monitor_script.py',
                                 bucket,
                                 '{}/scripts/postprocess_monitor_script.py'.format(prefix))
# Uploading inference evaluation script to S3
response = s3_client.upload_file('./scripts/inference_evaluation.py',
                                 bucket,
                                 '{}/scripts/inference_evaluation.py'.format(prefix))

## Create network config
network_config = NetworkConfig(
    security_group_ids=security_group_ids,
    subnets=subnets,
    enable_network_isolation=True,
    encrypt_inter_container_traffic=True,
)

## Create an instance of an SKLearnProcessor processor for use in the ProcessingStep
sklearn_processor = SKLearnProcessor(
    framework_version=framework_version,
    instance_type=processing_instance_type,
    instance_count=processing_instance_count,
    base_job_name="sklearn-feature-eng",
    role=role,
    output_kms_key = kms_key,
    volume_kms_key = kms_key,
    network_config = network_config,
    tags=tags
)

step_process_input_data = ProcessingStep(
    name="DataPreProcessingStep",
    processor=sklearn_processor,
    inputs=[
        ProcessingInput(source=input_data_path, destination="/opt/ml/processing/input"),
    ],
    outputs=[
        ProcessingOutput(output_name="scoring-input-processed", source="/opt/ml/processing/output", destination=f"s3://{bucket}/{prefix}/batch-scoring/data/processed")
    ],
    code=f"s3://{bucket}/{prefix}/scripts/scoring_preprocessing.py",
)

## Create Model Step
model = Model(
    image_uri=image_uri,
    model_data=model_artifact_uri,
    sagemaker_session=sagemaker_session,
    role=role,
    model_kms_key=kms_key,
    env={
         "SAGEMAKER_SUBMIT_DIRECTORY": sagemaker_submit_directory,
         "SAGEMAKER_PROGRAM": sagemaker_program # the entry point present in training-src-files.tar.gz
        },
)

inputs = CreateModelInput(
    instance_type=transform_instance_type
)
step_create_model = CreateModelStep(
    name="CreateModelStep",
    model=model,
    inputs=inputs,
)

## Transform Step to Perform Batch Transformation
transformer = Transformer(
    model_name=step_create_model.properties.ModelName,
    instance_type=transform_instance_type,
    instance_count=transform_instance_count,
    output_path=Join(
                        on="/",
                        values=[
                            scoring_results_path,
                            ExecutionVariables.PIPELINE_EXECUTION_ID,
                        ],
                    ),
    output_kms_key = kms_key,
    volume_kms_key = kms_key,
    tags=tags,
    max_payload=1,
    max_concurrent_transforms=1,
)

step_transform = TransformStep(
    name="BatchInferenceStep", 
    transformer=transformer, 
    inputs=TransformInput(data=step_process_input_data.properties.ProcessingOutputConfig.Outputs["scoring-input-processed"].S3Output.S3Uri,
                         data_type='S3Prefix',
                         content_type='text/csv',
                         split_type='Line')
)

## Prepare Data for Model Monitoring
data_check_preprocess = ScriptProcessor(
    image_uri=image_uri,
    command=["python3"],
    instance_type=processing_instance_type,
    instance_count=processing_instance_count,
    base_job_name="data-check-preprocess",
    role=role,
    output_kms_key = kms_key,
    volume_kms_key = kms_key,
    network_config = network_config,
    tags = tags,
)

step_process_data_quality = ProcessingStep(
    name="DataCheckPreProcessingStep",
    processor=data_check_preprocess,
    inputs=[
        ProcessingInput(source=step_transform.properties.TransformOutput.S3OutputPath, 
                        destination="/opt/ml/processing/inference_results"),
        ProcessingInput(source=step_process_input_data.properties.ProcessingOutputConfig.Outputs["scoring-input-processed"].S3Output.S3Uri, 
                        destination="/opt/ml/processing/inference_data")
    ],
    outputs=[
        ProcessingOutput(output_name="inference-data-combined", source="/opt/ml/processing/output", destination=f"s3://{bucket}/{prefix}/batch-scoring/combined")
    ],
    code=f"s3://{bucket}/{prefix}/scripts/data_check_preprocessing.py",
)

## Performing Data Quality check
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
    env = {
                "PipelineName": pipeline_name,
                "Region": region,
            }
)

data_quality_check_config = DataQualityCheckConfig(
    baseline_dataset=step_process_data_quality.properties.ProcessingOutputConfig.Outputs['inference-data-combined'].S3Output.S3Uri,
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
    post_analytics_processor_script=f"s3://{bucket}/{prefix}/scripts/postprocess_monitor_script.py"
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

## Performing Model Quality check
response = s3_client.upload_file('./data/ground-truth.csv', 
                                 bucket,
                                 '{}/batch-scoring/data/ground-truth/ground-truth.csv'.format(prefix))

model_quality_calc = ScriptProcessor(
    image_uri=image_uri,
    command=["python3"],
    instance_type=processing_instance_type,
    instance_count=processing_instance_count,
    base_job_name="model-quality-check-calc",
    role=role,
    output_kms_key = kms_key,
    volume_kms_key = kms_key,
    network_config = network_config,
    tags = tags,
)

step_model_quality_calc = ProcessingStep(
    name="ModelQualityCalcStep",
    processor=model_quality_calc,
    inputs=[
        ProcessingInput(source=step_transform.properties.TransformOutput.S3OutputPath, 
                        destination="/opt/ml/processing/inference_results"),
        ProcessingInput(source=f"s3://{bucket}/{prefix}/batch-scoring/data/ground-truth/ground-truth.csv",
                        destination="/opt/ml/processing/ground_truth")
    ],
    outputs=[
        ProcessingOutput(output_name="model-quality-report", source="/opt/ml/processing/output", 
                         destination=Join(
                                on="/",
                                values=[
                                    "s3:/",
                                    bucket,
                                    prefix,
                                    "monitoring",
                                    ExecutionVariables.PIPELINE_EXECUTION_ID,
                                    "modelqualitycheckstep",
                                ],
                            )
                        )
    ],
    code=f"s3://{bucket}/{prefix}/scripts/inference_evaluation.py",
    job_arguments = ["--mse-threshold", acc_mse_threshold],
)

## Define a Condition Step to Check Availability of GroundTruth Data and Conditionally Perform Model Quality Calculation
ground_truth_available = ParameterBoolean(name="ground_truth_available", default_value=True)

cond = ConditionEquals(
    left=ground_truth_available,
    right=True,
)

step_cond = ConditionStep(
    name="GroundTruthAvailableCond",
    conditions=[cond],
        if_steps=[step_model_quality_calc],
    else_steps=[],
)

# Create the pipeline object
pipeline = Pipeline(
    name=pipeline_name,
    parameters=[
        processing_instance_type,
        processing_instance_count,
        transform_instance_type,
        transform_instance_count,
        monitoring_instance_count,
        monitoring_instance_type,
        input_data_path,
        scoring_results_path,
        user_id,
        acc_mse_threshold,
        ground_truth_available,
        skip_check_data_quality,
        register_new_baseline_data_quality,
        supplied_baseline_statistics_data_quality,
        supplied_baseline_constraints_data_quality
    ],
    steps=[step_process_input_data, step_create_model, 
           step_process_data_quality, step_transform, data_quality_check_step, step_cond],
)

print(pipeline)

# Validate the pipeline definition
definition = json.loads(pipeline.definition())
print(definition)

# Update/create the pipeline
# If set, delete existing pipeline first
if recreate_pipelines == "true":
    try:
        sm_client.delete_pipeline(PipelineName=pipeline_name)
        print('Pipeline {} has been deleted'.format(pipeline_name))
    except:
        print('Pipeline {} does not exist'.format(pipeline_name))

pipeline.upsert(role_arn=role) 

# If set, upload inference data
if upload_inference_data == "true":
    print('Uploading inference data')
    # Upload inference data to S3
    sagemaker.s3.S3Uploader.upload(
        local_path='./data/inference-data.csv',
        desired_s3_uri=input_data_uri,
    )

print('Pipeline {} has been created'.format(pipeline_name))