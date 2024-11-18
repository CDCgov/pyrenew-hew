"""
Set up a multi-location, multi-date,
potentially multi-disease end to end
retrospective evaluation run for pyrenew-hew
on Azure Batch.
"""

import os
from datetime import datetime, timezone
from pathlib import Path

from pygit2 import Repository
from setup_prod_job import main

if __name__ == "__main__":
    current_datetime = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%SZ")
    current_branch = Path(Repository(os.getcwd()).head.name).stem
    locs_to_exclude = [  # keep CA, MN, SD
        "AS",
        "GU",
        "MO",
        "MP",
        "PR",
        "UM",
        "VI",
        "WY",
        "AK",
        "AL",
        "AR",
        "AZ",
        "CO",
        "CT",
        "DC",
        "DE",
        "FL",
        "GA",
        "HI",
        "IA",
        "ID",
        "IL",
        "IN",
        "KS",
        "KY",
        "LA",
        "MA",
        "MD",
        "ME",
        "MI",
        "MS",
        "MT",
        "NC",
        "ND",
        "NE",
        "NH",
        "NJ",
        "NM",
        "NV",
        "NY",
        "OH",
        "OK",
        "OR",
        "PA",
        "RI",
        "SC",
        "TN",
        "TX",
        "UT",
        "VA",
        "VT",
        "WA",
        "WI",
        "WV",
    ]
    main(
        job_id=current_datetime,
        pool_id="pyrenew-pool",
        diseases=["COVID-19", "Influenza"],
        container_image_name="pyrenew-hew",
        container_image_version=current_branch,
        excluded_locations=locs_to_exclude,
        test=True,
    )
