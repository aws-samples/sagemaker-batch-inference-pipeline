################################
# Pipeline notification Lambda #
################################
resource "aws_iam_role" "iam_for_lambda" {
  tags               = local.tags
  name               = "${var.notification_function_name}-role"
  assume_role_policy = <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Action": "sts:AssumeRole",
      "Principal": {
        "Service": "lambda.amazonaws.com"
      },
      "Effect": "Allow"
    }
  ]
}
EOF
}

resource "aws_lambda_function" "pipeline_notification_lambda" {
  filename         = "lambda_function_payload.zip"
  function_name    = var.notification_function_name
  role             = aws_iam_role.iam_for_lambda.arn
  handler          = "index.lambda_handler"
  source_code_hash = filebase64sha256("lambda_function_payload.zip")
  runtime          = "python3.9"
  reserved_concurrent_executions = 100
  environment {
    variables = {
      TOPIC_ARN = aws_sns_topic.pipeline_notification_topic.arn
    }
  }
  tags = local.tags
}

resource "aws_iam_role_policy_attachment" "lambda_notification" {
  role       = aws_iam_role.iam_for_lambda.name
  policy_arn = aws_iam_policy.lambda_notification_policy.arn
}

resource "aws_iam_policy" "lambda_notification_policy" {
  name        = "${var.notification_function_name}-policy"
  policy      = data.aws_iam_policy_document.lambda_notification_policy.json
  description = "The IAM policy for notification lambda function."
}

data "aws_iam_policy_document" "lambda_notification_policy" {
  statement {
    sid       = "PutLogPermission"
    effect    = "Allow"
    resources = ["arn:aws:logs:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:log-group:*:log-stream:*"]
    actions = [
      "logs:DescribeLogGroups",
    ]
  }
  statement {
    sid       = "DescribeLogPermission"
    effect    = "Allow"
    resources = ["arn:aws:logs:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:log-group:*"]
    actions = [
      "logs:CreateLogStream",
      "logs:CreateLogDelivery",
      "logs:CreateLogGroup",
      "logs:CreateLogStream",
      "logs:GetLogDelivery",
      "logs:PutLogEvents",
    ]
  }
  statement {
    sid       = "SNSPermission"
    effect    = "Allow"
    resources = [aws_sns_topic.pipeline_notification_topic.arn]
    actions = [
      "SNS:Publish"
    ]
  }
}
