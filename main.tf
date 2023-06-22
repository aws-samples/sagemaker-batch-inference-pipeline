data "aws_caller_identity" "current" {}
data "aws_region" "current" {}

# Terraform Backend Configuration is optional if you defined the backend in your runner.
terraform {
  backend "s3" {
    bucket = "batch-scoring-pipeline-tf-state"
    key    = "test/test_state"
    region = "us-east-1"
  }
}

locals {
  tags                    = jsondecode(file("${path.module}/tags.json"))
  email_address           = var.email_recipient
  notification_setup_list = jsonencode(var.custom_notification_config)
}