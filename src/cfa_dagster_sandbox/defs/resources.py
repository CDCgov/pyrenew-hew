import os

import dagster as dg
from dagster_azure.adls2 import (
    ADLS2DefaultAzureCredential,
    ADLS2PickleIOManager,
    ADLS2Resource,
)
from dagster_azure.blob import (
    AzureBlobStorageDefaultCredential,
    AzureBlobStorageResource,
)

# get the user from the environment, throw an error if variable is not set
user = os.environ["DAGSTER_USER"]
# this prefix allows each user to have their own Dagster assets
adls2_prefix = f"dagster-files/{user}/"


@dg.definitions
def resources() -> dg.Definitions:
    return dg.Definitions(
        resources={
            # This IOManager lets Dagster serialize asset outputs and store them in Azure to pass between assets
            "io_manager": ADLS2PickleIOManager(
                adls2_file_system="cfadagsterdev",
                adls2_prefix=adls2_prefix,
                adls2=ADLS2Resource(
                    storage_account="cfadagsterdev",
                    credential=ADLS2DefaultAzureCredential(kwargs={}),
                ),
                lease_duration=-1,  # unlimited lease for writing large files
            ),
            "azure_blob_storage": AzureBlobStorageResource(
                # account_url="cfaazurebatchprd.blob.core.windows.net",
                account_url="cs210032002a3162a1d.blob.core.windows.net",
                credential=AzureBlobStorageDefaultCredential(),
            ),
        }
    )
