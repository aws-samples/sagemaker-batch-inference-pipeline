###############################################
# EventBridge rule for pipeline status change #
###############################################
resource "aws_cloudwatch_event_rule" "pipeline_status_change" {
  name          = "sm-pipeline-status-${var.user_id}-${data.aws_region.current.name}"
  description   = "Detect the pipeline execution status change on specific pipeline "
  tags          = local.tags
  event_pattern = <<EOF
{
  "source": ["aws.sagemaker"],
  "detail-type": ["SageMaker Model Building Pipeline Execution Status Change"],
  "detail": {
    "pipelineArn": [
      "arn:aws:sagemaker:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:pipeline/${lower(var.pipeline_train)}",
      "arn:aws:sagemaker:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:pipeline/${lower(var.pipeline_inf)}",
      "arn:aws:sagemaker:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:pipeline/${lower(var.pipeline_trainwhpo)}"
      ]
  }
}
EOF
}

resource "aws_cloudwatch_event_target" "pipeline_status_email_recipients" {
  rule      = aws_cloudwatch_event_rule.pipeline_status_change.name
  target_id = "pipeline-status-email-${var.user_id}"
  arn       = "arn:aws:lambda:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:function:${var.notification_function_name}"
  input_transformer {
    input_paths = {
      detail = "$.detail"
    }
    input_template = <<EOF
{
  "source": "SageMaker_Pipeline",
  "pipeline_update": "SageMaker Model Building Pipeline Execution Status Change",
  "execution_details": <detail>
}
EOF
  }
}

resource "aws_lambda_permission" "execution_eventbridge" {
  statement_id  = "AllowExecutionFromEvents${var.user_id}"
  action        = "lambda:InvokeFunction"
  function_name = var.notification_function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.pipeline_status_change.arn
}

############################################
# EventBridge rule for training step change #
############################################
resource "aws_cloudwatch_event_rule" "pipeline_train_step_change" {
  name          = "sm-pipeline-train-step-${var.user_id}-${data.aws_region.current.name}"
  description   = "Detect the pipeline execution step change on specific pipeline "
  tags          = local.tags
  event_pattern = <<EOF
{
  "source": ["aws.sagemaker"],
  "detail-type": ["SageMaker Model Building Pipeline Execution Step Status Change"],
  "detail": {
    "currentStepStatus": [
      "Starting",
      "Succeeded",
      "Failed",
      "Stopping"
    ],
    "pipelineArn": [
      "arn:aws:sagemaker:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:pipeline/${lower(var.pipeline_train)}",
      "arn:aws:sagemaker:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:pipeline/${lower(var.pipeline_trainwhpo)}"
      ]
  }
}
EOF
}

resource "aws_cloudwatch_event_target" "pipeline_train_step_notification" {
  rule      = aws_cloudwatch_event_rule.pipeline_train_step_change.name
  target_id = "pipeline-step-train-email-${var.user_id}"
  arn       = "arn:aws:lambda:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:function:${var.notification_function_name}"
  input_transformer {
    input_paths = {
      detail = "$.detail"
    }
    input_template = <<EOF
{
  "source": "SageMaker_Pipeline",
  "pipeline_update": "SageMaker Model Building Pipeline Execution Step Status Change",
  "execution_details": <detail>,
  "notification_setup_list":${local.notification_setup_list}
}
EOF
  }
}

resource "aws_lambda_permission" "step_train_notification" {
  statement_id  = "AllowStepStatusFromEvents-train-${var.user_id}"
  action        = "lambda:InvokeFunction"
  function_name = var.notification_function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.pipeline_train_step_change.arn
}

############################################
# EventBridge rule for inference pipeline #
############################################
resource "aws_cloudwatch_event_rule" "pipeline_inf_step_change" {
  name        = "sm-pipeline-inf-step-${var.user_id}-${data.aws_region.current.name}"
  description = "Detect the pipeline execution step change on specific pipeline "

  event_pattern = <<EOF
{
  "source": ["aws.sagemaker"],
  "detail-type": ["SageMaker Model Building Pipeline Execution Step Status Change"],
  "detail": {
    "currentStepStatus": [
      "Starting",
      "Succeeded",
      "Failed",
      "Stopping"
    ],
    "pipelineArn": [
      "arn:aws:sagemaker:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:pipeline/${lower(var.pipeline_inf)}"
      ]
  }
}
EOF
}

resource "aws_cloudwatch_event_target" "pipeline_inf_step_notification" {
  rule      = aws_cloudwatch_event_rule.pipeline_inf_step_change.name
  target_id = "pipeline-step-inf-email-${var.user_id}"
  arn       = "arn:aws:lambda:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:function:${var.notification_function_name}"
  input_transformer {
    input_paths = {
      detail = "$.detail"
    }
    input_template = <<EOF
{
  "source": "SageMaker_Pipeline",
  "pipeline_update": "SageMaker Model Building Pipeline Execution Step Status Change",
  "execution_details": <detail>,
  "notification_setup_list":${local.notification_setup_list}
}
EOF
  }
}

resource "aws_lambda_permission" "step_inf_notification" {
  statement_id  = "AllowStepStatusFromEvents-inf-${var.user_id}"
  action        = "lambda:InvokeFunction"
  function_name = var.notification_function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.pipeline_inf_step_change.arn
}

##########################################################
# EventBridge rule for failed step - inference pipeline #
##########################################################
resource "aws_cloudwatch_event_rule" "pipeline_infer_step_failure" {
  name        = "sm-pipeline-fail-${var.pipeline_inf}-${var.user_id}-${data.aws_region.current.name}"
  description = "Detect a failure on the inference pipeleine's module quality calculation to trigger the training pipeline"
  # The step name of the AWS Model Building Pipeline is ModelQualityCalcStep
  event_pattern = <<EOF
{
  "source": ["aws.sagemaker"],
  "detail-type": ["SageMaker Model Building Pipeline Execution Step Status Change"],
  "detail": {
    "pipelineArn": ["arn:aws:sagemaker:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:pipeline/${lower(var.pipeline_inf)}"],
    "stepName": ["ModelQualityCalcStep"],
    "currentStepStatus": ["Failed"]
  }
}
EOF
}

resource "aws_cloudwatch_event_target" "pipeline_infer_sagemaker_pipeline" {
  rule      = aws_cloudwatch_event_rule.pipeline_infer_step_failure.name
  target_id = "trigger_sm_pipeline-${var.pipeline_inf}-${var.user_id}"
  arn       = "arn:aws:sagemaker:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:pipeline/${lower(var.pipeline_trainwhpo)}"
  role_arn  = aws_iam_role.events_role.arn
}

###################################
# EventBridge IAM Role and Policy #
###################################
resource "aws_iam_role" "events_role" {
  name               = "sm-pipeline-events-role-${var.user_id}-${data.aws_region.current.name}"
  description        = "IAM Role to allow events to trigger sagemaker pipeline."
  tags               = local.tags
  assume_role_policy = data.aws_iam_policy_document.events_trust_policy.json
}

resource "aws_iam_policy" "events_policy" {
  name        = "sm-pipeline-events-policy-${var.user_id}-${data.aws_region.current.name}"
  policy      = data.aws_iam_policy_document.events_policy.json
  description = "The IAM policy allow CloudWatch events to trigger sagemaker pipeline."
}

resource "aws_iam_role_policy_attachment" "events_role_events_policy" {
  role       = aws_iam_role.events_role.name
  policy_arn = aws_iam_policy.events_policy.arn
}

data "aws_iam_policy_document" "events_trust_policy" {
  statement {
    effect = "Allow"
    actions = [
      "sts:AssumeRole",
    ]
    principals {
      type = "Service"
      identifiers = [
        "events.amazonaws.com",
      ]
    }
  }
}

data "aws_iam_policy_document" "events_policy" {
  statement {
    sid       = "DescribeLogGroup"
    effect    = "Allow"
    resources = ["arn:aws:logs:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:log-group:*:log-stream:*"]
    actions = [
      "logs:DescribeLogGroups",
    ]
  }
  statement {
    sid       = "AccessLogPermission"
    effect    = "Allow"
    resources = ["arn:aws:logs:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:log-group:*"]
    actions = [
      "logs:CreateLogStream",
      "logs:CreateLogDelivery",
      "logs:CreateLogGroup",
      "logs:CreateLogStream",
      "logs:GetLogDelivery",
      "logs:DescribeLogGroups",
      "logs:DescribeLogStreams",
      "logs:PutRetentionPolicy",
      "logs:GetLogDelivery",
      "logs:PutLogEvents",
      "logs:GetLogEvents",
    ]
  }
  statement {
    sid    = "SNSPermission"
    effect = "Allow"
    actions = [
      "sns:*",
    ]
    resources = [
      "arn:aws:sns:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:*",
    ]
  }
  statement {
    sid    = "SMPipelinePermission"
    effect = "Allow"
    actions = [
      "sagemaker:StartPipelineExecution",
    ]
    resources = [
      "arn:aws:sagemaker:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:pipeline/${lower(var.pipeline_train)}",
      "arn:aws:sagemaker:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:pipeline/${lower(var.pipeline_inf)}",
      "arn:aws:sagemaker:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:pipeline/${lower(var.pipeline_trainwhpo)}",
    ]
  }
}

###########################
# S3 Oject Event Triggers #
###########################
resource "aws_cloudwatch_event_rule" "s3_trigger_inf_pipeline" {
  name        = "sm-pipeline-s3event-${var.pipeline_inf}-${var.user_id}"
  description = "s3 event to trigger the inference pipeline"
  tags        = local.tags
  # This event example monitors the file "inference-data.csv".
  event_pattern = <<EOF
{
    "source": ["aws.s3"],
    "detail-type": ["AWS API Call via CloudTrail"],
    "detail": {
        "eventSource": ["s3.amazonaws.com"],
        "eventName": ["PutObject"],
        "requestParameters": {
            "bucketName": ["${var.bucket_name}"],
            "key": [
              {"prefix": "${var.bucket_prefix}/${var.bucket_inf_prefix}/inference-data.csv"}
              ]
        }
    }
}
EOF
}

resource "aws_cloudwatch_event_target" "pipeline_infer_s3event_pipeline" {
  rule      = aws_cloudwatch_event_rule.s3_trigger_inf_pipeline.name
  target_id = "trigger_sm_s3event-${var.pipeline_inf}-${var.user_id}"
  arn       = "arn:aws:sagemaker:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:pipeline/${lower(var.pipeline_inf)}"
  role_arn  = aws_iam_role.events_role.arn
}

resource "aws_cloudwatch_event_rule" "s3_trigger_train_pipeline" {
  name        = "sm-pipeline-s3event-${var.pipeline_train}-${var.user_id}"
  description = "s3 event to trigger the training pipeline"
  tags        = local.tags
  # This event example monitors the file "training-data.csv".
  event_pattern = <<EOF
{
    "source": ["aws.s3"],
    "detail-type": ["AWS API Call via CloudTrail"],
    "detail": {
        "eventSource": ["s3.amazonaws.com"],
        "eventName": ["PutObject"],
        "requestParameters": {
            "bucketName": ["${var.bucket_name}"],
            "key": [
              {"prefix": "${var.bucket_prefix}/${var.bucket_train_prefix}/training-data.csv"}
              ]
        }
    }
}
EOF
}

resource "aws_cloudwatch_event_target" "pipeline_train_s3event_pipeline" {
  rule      = aws_cloudwatch_event_rule.s3_trigger_train_pipeline.name
  target_id = "trigger_sm_s3event-${var.pipeline_train}-${var.user_id}"
  arn       = "arn:aws:sagemaker:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:pipeline/${lower(var.pipeline_train)}"
  role_arn  = aws_iam_role.events_role.arn
}