"""Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved"""
import os
import json
import logging
import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger()
logger.setLevel(logging.INFO)
sns_client = boto3.client('sns')

TOPIC_ARN = os.environ['TOPIC_ARN']


def sm_pipeline_exec_notification(pipeline_event):
    """To send notification email for FAILED sagemaker pipeline execution."""

    execution_details = pipeline_event["execution_details"]
    exec_status = execution_details["currentPipelineExecutionStatus"]
    pipeline_name = execution_details["pipelineArn"].split("/")[-1]
    email_subject = f"SageMaker Pipeline: {pipeline_name} Execution - {exec_status}"  # noqa: E501

    try: 
        sns_response = sns_client.publish(
                        TopicArn=TOPIC_ARN,
                        Subject=email_subject,
                        Message=json.dumps(exec_status,sort_keys=True, indent=4)
                    )
        logger.info("Email of pipeline failure was sent successfully. %s",sns_response)

    except Exception as error:
        logger.error(error)


def sm_pipeline_step_notification(pipeline_event):
    """To send notification email for SELECTED pipeline steps."""
    execution_details = pipeline_event["execution_details"]
    logger.debug(execution_details.items())
    step_name = execution_details["stepName"]
    current_step_status = execution_details["currentStepStatus"]
    addon_msg = "None"
    email_subject = f"SageMaker Pipeline Step: {step_name} Execution - {current_step_status}"  # noqa: E501

    ###############################################################
    # Adding extra custom message for predefined steps and status #
    ###############################################################
    if "notification_setup_list" in pipeline_event:
        for custom_notification in pipeline_event["notification_setup_list"]:  # noqa: E501
            if custom_notification["step_name"] == step_name and custom_notification["step_status"]  == current_step_status:  # noqa: E501
                addon_msg = custom_notification["step_msg"]
                email_subject += " - Action Required"

    execution_details["Addon_Message"]=str(addon_msg)
    try: 
        sns_response = sns_client.publish(
                        TopicArn=TOPIC_ARN,
                        Subject=email_subject,
                        Message=json.dumps(execution_details, sort_keys=True, indent=4)
                    )

        logger.info("Email of pipeline step change was sent successfully.\n %s",sns_response)
    except Exception as error:
        logger.error(error)


def processing_sm_pipeline_notification(event):
    """To send notification email based on the pipeline execution types."""
    if "Execution Step Status Change" in event["pipeline_update"]:
        logger.info("Pipeline Step Status Notification")
        sm_pipeline_step_notification(event)

    elif "Pipeline Execution Status Change" in event["pipeline_update"]:
        logger.info("Pipeline Execution Status Notification")
        sm_pipeline_exec_notification(event)
    else:
        logger.error("Invalid pipeline update event.")


def lambda_handler(event, context):
    """Lambda_handler"""
    logger.info(event)
    logger.debug(context)

    try:
        if "pipeline_update" in event:
            logger.info("Received pipeline status notification")
            processing_sm_pipeline_notification(event)

    except KeyError as client_error:
        logger.error("The event trigger: %s is invalid!", client_error)
        return False
