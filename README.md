# PyRenew-HEW
| hewr | pyrenew-hew | pipelines |
| ---- | ----------- | --------- |
| [![hewr](https://codecov.io/gh/CDCgov/pyrenew-hew/branch/main/graph/badge.svg?flag=hewr)](https://codecov.io/gh/CDCgov/pyrenew-hew) | [![pyrenew_hew](https://codecov.io/gh/CDCgov/pyrenew-hew/graph/badge.svg?flag=pyrenew_hew)](https://codecov.io/gh/CDCgov/pyrenew-hew) | [![pipelines](https://codecov.io/gh/CDCgov/pyrenew-hew/graph/badge.svg?flag=pipelines)](https://codecov.io/gh/CDCgov/pyrenew-hew) |

The PyRenew-HEW project aims to create short-term forecasts of respiratory disease burden using the [PyRenew library](https://github.com/CDCgov/PyRenew) and several data sources:

- [x] **H**ospital Admissions from the [National Healthcare Safety Network](https://www.cdc.gov/nhsn/index.html)
- [x] **E**mergency Department Visits from the [National Syndromic Surveillance Program](https://www.cdc.gov/nssp/index.html)
- [x] **W**astewater virus concentration from the [National Wastewater Surveillance System](https://www.cdc.gov/nwss/index.html)

This is a work in progress, and not all data sources are currently integrated into the model.

This repository contains code for the [PyRenew-HEW model](https://github.com/CDCgov/pyrenew-hew/blob/main/pyrenew_hew/pyrenew_hew_model.py) itself, as well as [pipelines](https://github.com/CDCgov/pyrenew-hew/tree/main/pyrenew_hew) for running the model in production, and [utilities](https://github.com/CDCgov/pyrenew-hew/tree/main/hewr) for working with model outputs.

## Containers

### Standard Container
The project uses GitHub Actions for automatically building container images based on the project's [Containerfile](Containerfile). The images are currently hosted on Github Container Registry and are built and pushed via the [containers.yaml](.github/workflows/containers.yaml) GitHub Actions workflow.

Images can also be built locally. The [Makefile](Makefile) contains several targets for building and pushing images. Although the Makefile uses Docker as the default engine, the `ENGINE` environment variable can be set to `podman` to use Podman instead, for example:

```bash
ENGINE=podman make container_build
# Equivalent to:
# podman build . -t pyrenew-hew -f Containerfile
```

Container images pushed to the Azure Container Registry are automatically tagged as either `latest` (if the commit is on the `main` branch) or with the branch name (if the commit is on a different branch). After a branch is deleted, the image tag is remove from the registry via the [delete-container-tag.yaml](.github/workflows/delete-container-tag.yaml) GitHub Actions workflow.

## PyRenew Dagster Sandbox
`dagster_defs.py` can be used to launch a dagster version of the batch pipeline specified in the `setup_job.py` script.

1. If you have never set up Dagster on your VAP before, you will need to set up a `~/.dagster_home/dagster.yaml` file: `uv run https://raw.githubusercontent.com/CDCgov/cfa-dagster/refs/heads/main/setup.py`
2. Build the initial image for your test asset: `docker build -t pyrenew-dagster -f Containerfile .`
3. Start the Dagster UI by running `uv run dagster_defs.py --dev` and clicking the link in your terminal (usually [http://127.0.0.1:3000/])
4. Materialize an asset!

### Next Steps
1. Push your updated image to ACR: 
    - `az login --identity && az acr login -n cfaprdbatchcr && docker build -t cfaprdbatchcr.azurecr.io/pyrenew-dagster:$(basename $HOME) . --push`
2. Modify the `dagster_defs.py` file to use the `azure_caj_executor` or `azure_batch_executor` instead of the `docker_executor`
4. Materialize your Asset again! (See `Getting Started`)

## Running Model Pipelines
> [!NOTE]
>
> Please contact a project maintainer if you believe you need access to pipelines.
> CDC internal users can check regularly scheduled jobs at [PyRenew-Cron](https://github.com/cdcent/pyrenew-cron).

> Specific environment setup steps required can be found in the [Routine Forecasting Standard Operating Procedure](https://cdcent.github.io/cfa-stf-handbook/routine_forecast_sop.html).

> Additionally, pipelines as written can only be run within CFA's Virtual Analyst Platform.

Pipelines can be run interactively or non-interactively:
- `pipelines/azure_command_center.py` is now the preferred method of running model fit pipeline jobs interactively.
- The `Makefile` also provides targets that will run pipelines non-interactively. Run `make help` for more information.
- Pipelines are run through the command line python interface when scheduled using [Pyrenew-Cron](https://github.com/cdcent/pyrenew-cron).

## General Disclaimer
This repository was created for use by CDC programs to collaborate on public health related projects in support of the [CDC mission](https://www.cdc.gov/about/organization/mission.htm).  GitHub is not hosted by the CDC, but is a third party website used by CDC and its partners to share information and collaborate on software. CDC use of GitHub does not imply an endorsement of any one particular service, product, or enterprise.

## Public Domain Standard Notice
This repository constitutes a work of the United States Government and is not
subject to domestic copyright protection under 17 USC § 105. This repository is in
the public domain within the United States, and copyright and related rights in
the work worldwide are waived through the [CC0 1.0 Universal public domain dedication](https://creativecommons.org/publicdomain/zero/1.0/).
All contributions to this repository will be released under the CC0 dedication. By
submitting a pull request you are agreeing to comply with this waiver of
copyright interest.

## License Standard Notice
This repository is licensed under ASL v2 or later.

This source code in this repository is free: you can redistribute it and/or modify it under
the terms of the Apache Software License version 2, or (at your option) any
later version.

This source code in this repository is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the Apache Software License for more details.

You should have received a copy of the Apache Software License along with this
program. If not, see http://www.apache.org/licenses/LICENSE-2.0.html

The source code forked from other open source projects will inherit its license.

## Privacy Standard Notice
This repository contains only non-sensitive, publicly available data and
information. All material and community participation is covered by the
[Disclaimer](https://github.com/CDCgov/template/blob/master/DISCLAIMER.md)
and [Code of Conduct](https://github.com/CDCgov/template/blob/master/code-of-conduct.md).
For more information about CDC's privacy policy, please visit [http://www.cdc.gov/other/privacy.html](https://www.cdc.gov/other/privacy.html).

## Contributing Standard Notice
Anyone is encouraged to contribute to the repository by [forking](https://help.github.com/articles/fork-a-repo)
and submitting a pull request. (If you are new to GitHub, you might start with a
[basic tutorial](https://help.github.com/articles/set-up-git).) By contributing
to this project, you grant a world-wide, royalty-free, perpetual, irrevocable,
non-exclusive, transferable license to all users under the terms of the
[Apache Software License v2](http://www.apache.org/licenses/LICENSE-2.0.html) or
later.

All comments, messages, pull requests, and other submissions received through
CDC including this GitHub page may be subject to applicable federal law, including but not limited to the Federal Records Act, and may be archived. Learn more at [http://www.cdc.gov/other/privacy.html](http://www.cdc.gov/other/privacy.html).

## Records Management Standard Notice
This repository is not a source of government records but is a copy to increase
collaboration and collaborative potential. All government records will be
published through the [CDC web site](http://www.cdc.gov).
