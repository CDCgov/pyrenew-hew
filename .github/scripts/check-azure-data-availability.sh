#!/bin/bash

# This script checks the availability of data in Azure Blob Storage for various datasets.
# It compares the last modified date of the data with the current date in UTC.
current_date=$(date -u +%Y-%m-%d)

# To expose variables to the parent shell, source this script instead of executing it.
# Usage: source /home/qxk3/repos/pyrenew-hew/.github/scripts/check-data-availability.sh

# The variables nssp_gold_check, nwss_vintages_check, and nhsn_check will then be available in the parent shell.
# nssp
echo "-----------------------------------------"
echo "Checking nssp gold data availability..."
echo "-----------------------------------------"
nssp_gold_last_modified=$(
	az storage blob list \
	--account-name "cfaazurebatchprd" \
	--container-name "nssp-etl" \
	--prefix "gold/" \
	--query "sort_by([?ends_with(name, '.parquet')], &properties.lastModified)[-1].properties.lastModified" \
	--auth-mode login \
	--output tsv \
	| cut -d 'T' -f 1 \
	| date -u -d "$1" +%Y-%m-%d
)

echo "-----------------------------------------"
echo "Current date in UTC:"
echo "$current_date"
echo "Latest nssp gold date:"
echo "$nssp_gold_last_modified"
echo "-----------------------------------------"

if [[ ! "$nssp_gold_last_modified" < "$current_date" ]]; then
	nssp_gold_check=true
	echo "The nssp data is up to date with today."
else
	nssp_gold_check=false
	echo "The nssp data is older than the current date."
fi
echo "-----------------------------------------"

# nwss
echo "-----------------------------------------"
echo "Checking nwss-vintages data availability..."
echo "-----------------------------------------"

nwss_vintages_folder="NWSS-ETL-covid-$current_date/"
nwss_vintages_exists=$(
	az storage blob list \
		--account-name "cfaazurebatchprd" \
		--container-name "nwss-vintages" \
		--prefix "$nwss_vintages_folder" \
		--auth-mode login \
		--query "[0].name" \
		--output tsv
)

echo "Looking for folder: $nwss_vintages_folder"
if [[ -n "$nwss_vintages_exists" ]]; then
	nwss_vintages_check=true
	echo "The NWSS-ETL-covid-$current_date folder exists."
else
	nwss_vintages_check=false
	echo "The NWSS-ETL-covid-$current_date folder does not exist."
fi
echo "-----------------------------------------"

# nhsn
echo "-----------------------------------------"
echo "Checking NHSN API data availability..."
echo "-----------------------------------------"

nhsn_target_url="https://data.cdc.gov/api/views/mpgq-jmmr.json"
nhsn_update_date_raw=$(curl -sf "$nhsn_target_url" | jq -r '.rowsUpdatedAt // empty')

if [[ -z "$nhsn_update_date_raw" ]]; then
	nhsn_check=false
	echo "Key 'rowsUpdatedAt' not found in NHSN API response."
else
	nhsn_update_date=$(date -u -d @"$nhsn_update_date_raw" +%Y-%m-%d 2>/dev/null)
	if [[ -z "$nhsn_update_date" ]]; then
		nhsn_check=false
		echo "Error processing NHSN API response: Invalid timestamp."
	else
		echo "NHSN last update date: $nhsn_update_date"
		if [[ "$nhsn_update_date" == "$current_date" ]]; then
			nhsn_check=true
			echo "The NHSN data is up to date with today."
		else
			nhsn_check=false
			echo "The NHSN data is not up to date with today."
		fi
	fi
fi

echo "========================================="
echo "Boolean check results:"
echo "nssp_gold_check: $nssp_gold_check"
echo "nwss_vintages_check: $nwss_vintages_check"
echo "nhsn_check: $nhsn_check"
echo "========================================="
