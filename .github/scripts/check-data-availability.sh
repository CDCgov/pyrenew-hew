# nssp
nssp_gold_last_modified=$(
	az storage blob show \
		--account-name "cfaazurebatchprd" \
		--container-name "nssp-etl" \
		--name "gold.parquet" \
		--query "properties.lastModified" \
		--auth-mode login \
		--output tsv
)

# convert to a simple date
raw_nssp_gold_date=$(echo $nssp_gold_last_modified | cut -d 'T' -f 1)

# NSSP in utc time
nssp_gold_date=$(date -u -d "$raw_nssp_gold_date" +%Y-%m-%d)
current_date=$(date -u +%Y-%m-%d)

echo "-----------------------------------------"
echo "Latest gold last modified date:"
echo $nssp_gold_date
echo "-----------------------------------------"

# TODO: logic to compare nssp_gold_date with current date
# TODO: create boolean output variable for the check that we can check in later job
# We modify both to fit utc timezone

if [[ "$utc_nssp_gold_date" < "$current_date" ]]; then
	echo "nssp_gold_check=true" >>$GITHUB_OUTPUT
else
	echo "nssp_gold_check=false" >>$GITHUB_OUTPUT
fi

# timeseries-e
# TODO: check test output too, if the flag is set
# timseries_e_output=$(
#   az storage blob show \
#   --account-name "cfaazurebatchprd" \
#   --container-name "pyrenew-hew-prod-output" \
#   --name "" \
#   --query "properties.lastModified" \
#   --auth-mode login \
#   --output tsv
# )

# TODO: create boolean output variable for the check that we can check in later job
if [[ "$timseries_e_output" < "$current_date" ]]; then
	echo "timseries_e_check=true" >>$GITHUB_OUTPUT
else
	echo "timseries_e_check=false" >>$GITHUB_OUTPUT
fi

# nwss
# nwss_gold_last_modified=$(
#   az storage blob show \
#   --account-name "cfaazurebatchprd" \
#   --container-name "nssp-etl" \
#   --name "latest_comprehensive.parquet" \
#   --query "properties.lastModified" \
#   --auth-mode login \
#   --output tsv
# )

# TODO: create boolean output variable for the check that we can check in later job
if [[ "$nwss_gold_last_modified" < "$current_date" ]]; then
	echo "nwss_gold_check=true" >>$GITHUB_OUTPUT
else
	echo "nwss_gold_check=false" >>$GITHUB_OUTPUT
fi

# nhsn
# TODO: nshn api code
