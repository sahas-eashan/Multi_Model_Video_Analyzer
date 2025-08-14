from sagemaker.pytorch import PyTorch
from sagemaker.debugger import TensorBoardOutputConfig


bucket = "sentimenttanalysis1"  # <-- your bucket (two t’s)
prefix = "sagemaker"


def start_training():
    tensor_board_config = TensorBoardOutputConfig(
        s3_output_path=f"s3://{bucket}/tensorboard",
        container_local_output_path="/opt/ml/output/tensorboard",
    )

    estimator = PyTorch(
        entry_point="train.py",
        source_dir="training",
        role="arn:aws:iam::876042377496:role/sentiment-analysis-ex-role",
        framework_version="2.5.1",
        py_version="py311",
        instance_count=1,
        instance_type="ml.m5.large",
        hyperparameters={
            "batch-size": 8,
            "epochs": 25,
            "gradient-accumulation-steps": 4,
        },
        tensorboard_config=tensor_board_config,
        output_path=f"s3://{bucket}/{prefix}/outputs",  # <-- avoids default bucket
        code_location=f"s3://{bucket}/{prefix}/code",
    )

    # start training
    estimator.fit(
        {
            "training": "s3://sentimenttanalysis1/dataset/train",
            "validation": "s3://sentimenttanalysis1/dataset/dev",
            "test": "s3://sentimenttanalysis1/dataset/test",
        }
    )


if __name__ == "__main__":
    start_training()
