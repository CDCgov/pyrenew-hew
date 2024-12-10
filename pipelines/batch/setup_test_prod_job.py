"""
Set up a multi-location, multi-date,
potentially multi-disease end to end
retrospective evaluation run for pyrenew-hew
on Azure Batch.
"""

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path

from pygit2 import Repository
from setup_prod_job import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test production pipeline on small subset of locations"
    )
    parser.add_argument(
        "--tag",
        type=str,
        help="The tag name to use for the container image version",
        default=Path(Repository(os.getcwd()).head.name).stem,
    )

    args = parser.parse_args()

    tag = args.tag
    print(f"Using tag {tag}")
    current_datetime = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%SZ")
    tag = Path(Repository(os.getcwd()).head.name).stem

    locs_to_exclude = [  # keep CA, MN, SD, and US
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
        job_id=f"pyrenew-hew-test-{current_datetime}",
        pool_id="dmb-pyrenew-pool",
        diseases=["COVID-19", "Influenza"],
        container_image_name="pyrenew-hew",
        container_image_version=tag,
        excluded_locations=locs_to_exclude,
        test=True,
    )
