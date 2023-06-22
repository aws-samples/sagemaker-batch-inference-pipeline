bucket_name         =       # e.g. "sagemaker-bucket-abcdefg12345"
bucket_prefix       =       # e.g. "pipeline_shared/batch-scoring"
bucket_train_prefix =       # e.g. "training/data/raw"
bucket_inf_prefix   =       # e.g. "batch-scoring/data/raw"
email_recipient     =       # e.g. ["recipient-1@example.com", "recipient-2@example.com"]
user_id             =       # e.g.  "a12345"

custom_notification_config = [
  {
    step_name           =   # e.g. "ModelRegisterStep"
    step_status         =   # e.g. "Succeeded"
    step_custom_message =   # e.g. "This is the custom message for Succeeded \"ModelRegisterStep\" step."
  },
  {
    step_name           =   # e.g. "TrainingEval"
    step_status         =   # e.g. "Failed"
    step_custom_message =   # e.g. "This is the custom message for Failed \"TrainingEval\" step."
  }
]

# Pipeline information. Will be used by the Python helper script.
accuracy_mse_threshold     =   # e.g. "10.0"
kms_key                    =   # e.g. "arn:aws:kms:us-east-1:112233445566:key/123456a1-12b1-123c-1234-12345df12e12"
model_package_group_name =     # e.g. "poc-mpg"
notification_function_name =   # e.g. "pipeline-notification-lambda"
pipeline_inf               =   # e.g. "InferencePipeline"
pipeline_train             =   # e.g. "TrainingPipeline"
pipeline_trainwhpo         =   # e.g. "TrainingWHPOPipeline"

recreate_pipelines         =   # e.g. "true"
role_arn                   =   # e.g. "arn:aws:iam::112233445566:role/112233445566/sm_execution_role_batch_scoring"
sg_id                      =   # e.g. "sg-0a12b3c45b67de8f9"
subnet_id                  =   # "subnet-01a23bcdef45ghij6"
upload_inference_data      =   # e.g. "false"
upload_training_data       =   # e.g. "true"
