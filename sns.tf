##################################
# SNS resources for notification #
##################################
resource "aws_sns_topic" "pipeline_notification_topic" {
  name              = "sm-pipeline-notification-topic"
  kms_master_key_id = "alias/aws/sns" # AWS Default KMS Key
  tags              = local.tags
}

resource "aws_sns_topic_policy" "sns_topic_policy" {
  arn    = aws_sns_topic.pipeline_notification_topic.arn
  policy = data.aws_iam_policy_document.default_topic_policy.json
}

data "aws_iam_policy_document" "default_topic_policy" {
  statement {
    actions = [
      "sns:AddPermission",
      "sns:DeleteTopic",
      "sns:GetTopicAttributes",
      "sns:ListSubscriptionsByTopic",
      "sns:Publish",
      "sns:Receive",
      "sns:RemovePermission",
      "sns:SetTopicAttributes",
      "sns:Subscribe",
    ]
    condition {
      test     = "StringEquals"
      variable = "AWS:SourceOwner"
      values   = ["${data.aws_caller_identity.current.account_id}"]
    }
    effect = "Allow"
    principals {
      type        = "AWS"
      identifiers = ["*"]
    }
    resources = ["${aws_sns_topic.pipeline_notification_topic.arn}"]
  }
}

resource "aws_sns_topic_subscription" "topic_email_subscription" {
  count     = length(local.email_address)
  topic_arn = aws_sns_topic.pipeline_notification_topic.arn
  protocol  = "email-json"
  endpoint  = local.email_address[count.index]
}