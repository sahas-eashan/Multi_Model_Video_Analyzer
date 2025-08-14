import boto3
import os
from pathlib import Path


def deploy_to_aws(model_dir: str, bucket_name: str, endpoint_name: str):
    """Deploy saved model to AWS SageMaker"""

    sagemaker = boto3.client("sagemaker")
    s3 = boto3.client("s3")

    # Upload model artifacts to S3
    model_path = Path(model_dir)
    s3_prefix = "production-models"

    print("Uploading model to S3...")
    for file in model_path.glob("*"):
        s3.upload_file(str(file), bucket_name, f"{s3_prefix}/{file.name}")

    # Create SageMaker model
    role = "arn:aws:iam::876042377496:role/sentiment-analysis-ex-role"

    print("Creating SageMaker model...")
    sagemaker.create_model(
        ModelName=endpoint_name,
        ExecutionRoleArn=role,
        PrimaryContainer={
            "Image": f'{boto3.client("sts").get_caller_identity()["Account"]}.dkr.ecr.{boto3.Session().region_name}.amazonaws.com/sentiment-analyzer:latest',
            "ModelDataUrl": f"s3://{bucket_name}/{s3_prefix}/model.pth",
        },
    )

    # Create endpoint configuration
    print("Creating endpoint configuration...")
    sagemaker.create_endpoint_config(
        EndpointConfigName=endpoint_name,
        ProductionVariants=[
            {
                "VariantName": "AllTraffic",
                "ModelName": endpoint_name,
                "InstanceType": "ml.t2.medium",
                "InitialInstanceCount": 1,
            }
        ],
    )

    # Create endpoint
    print("Creating endpoint...")
    sagemaker.create_endpoint(
        EndpointName=endpoint_name, EndpointConfigName=endpoint_name
    )

    print(f"Endpoint {endpoint_name} is being created...")
    print("This may take a few minutes.")


if __name__ == "__main__":
    deploy_to_aws(
        model_dir="saved_models",
        bucket_name="sentimenttanalysis1",
        endpoint_name="sentiment-analyzer-endpoint",
    )
