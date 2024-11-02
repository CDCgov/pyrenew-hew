import argparse
import os

from azure.batch import models
from azuretools.auth import EnvCredentialHandler
from azuretools.client import get_batch_service_client
from azuretools.job import create_job_if_not_exists
from azuretools.task import get_container_settings, get_task_config


def main(job_id, pool_id, container_image, local_output_dir) -> None:
    creds = EnvCredentialHandler()
    client = get_batch_service_client(creds)
    job = models.JobAddParameter(
        id=job_id,
        pool_info=models.PoolInformation(pool_id=pool_id),
    )
    create_job_if_not_exists(client, job, verbose=True)

    container_settings = get_container_settings(
        container_image,
        working_directory="containerImageDefault",
        mount_pairs=[
            {
                "source": "nssp-etl",
                "target": "/pyrenew-hew/nssp_demo/nssp-etl",
            },
            {
                "source": "nssp-archival-vintages",
                "target": "/pyrenew-hew/nssp_demo/nssp-archival-vintages",
            },
            {
                "source": "prod-param-estimates",
                "target": "/pyrenew-hew/nssp_demo/params",
            },
            {
                "source": "pyrenew-test-output",
                "target": "/pyrenew-hew/nssp_demo/private_data",
            },
        ],
    )

    base_call = (
        "/bin/bash nssp_demo/loop_postprocess.sh "
        "nssp_demo/private_data/{dir_to_postprocess}"
    )

    dirs_to_postprocess = [
        item
        for item in os.listdir(local_output_dir)
        if item.startswith("influenza_r")
    ]

    for dir_name in dirs_to_postprocess:
        task = get_task_config(
            f"postprocess-{dir_name}",
            base_call=base_call.format(dir_to_postprocess=dir_name),
            container_settings=container_settings,
        )
        client.task.add(job_id, task)
    return None


parser = argparse.ArgumentParser()

parser.add_argument("job_id")
parser.add_argument("pool_id")
parser.add_argument("container_image")
parser.add_argument("local_output_dir")

if __name__ == "__main__":
    args = parser.parse_args()
    main(
        args.job_id, args.pool_id, args.container_image, args.local_output_dir
    )
