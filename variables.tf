variable "accuracy_mse_threshold" {
  type        = number
  description = "Maximum value for mse before requiring an update to the model"
  default     = 10.0
}

variable "bucket_name" {
  type        = string
  description = "The existing s3 bucket that is used to store the training data"
}

variable "bucket_prefix" {
  type        = string
  description = "S3 srefix for all s3 data"
}

variable "bucket_inf_prefix" {
  type        = string
  description = "S3 prefix for inference data"
}

variable "bucket_train_prefix" {
  type        = string
  description = "S3 prefix for training data"
}

variable "custom_notification_config" {
  description = "The custom notification message for specific SageMaker Model Building Pipeline steps with specific execution status"
  type = list(object({
    step_name           = string
    step_status         = string
    step_custom_message = string
  }))
}

variable "email_recipient" {
  type        = list(string)
  description = "The email address list for receiving Sagemaker pipeline notification messages"
}

variable "kms_key" {
  type        = string
  description = "KMS key ARN for s3 and SageMaker pipeline encryption"
}

variable "model_package_group_name" {
  type        = string
  description = "Name of the model used in the SageMaker pipelines"
}

variable "notification_function_name" {
  type        = string
  description = "Name of the Sagemaker domain module email notification lambda function"
  default     = "pipeline-notification-lambda"
}

variable "pipeline_trainwhpo" {
  type        = string
  description = "Name of training pipeline with hyper parameter optimization"
  default     = "TrainingWHPOPipeline"
}

variable "pipeline_train" {
  type        = string
  description = "The name of Sagemaker training pipeline"
  default     = "TrainingPipeline"
}

variable "pipeline_inf" {
  type        = string
  description = "The name of Sagemaker inference pipeline"
  default     = "InferencePipeline"
}

variable "recreate_pipelines" {
  type        = string
  description = "If set to true, any existing Sagemaker Pipelines will be deleted prior to creation"
  default     = true
}
variable "role_arn" {
  type        = string
  description = "The IAM resource arn of the Sagemaker pipeline role"
}

variable "upload_training_data" {
  type        = bool
  description = "If set to true, training data will be uploaded to s3 and this upload operation will trigger the training pipeline to execute"
  default     = true
}
variable "upload_inference_data" {
  type        = bool
  description = "If set to true, inference data will be uploaded to s3 and this upload operation will trigger the inference pipeline to execute"
  default     = false
}
variable "user_id" {
  type        = string
  description = "The id of current SageMaker user, eg a12345. This can be used to distinguish between different users"
}

variable "tags" {
  type        = map(string)
  description = "Resource Tags"
  default     = {}
}