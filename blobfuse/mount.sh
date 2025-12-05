#!/bin/bash

# ensure logged in via Azure CLI.
./blobfuse/verifylogin.sh

# pull azure configuration files
./blobfuse/pull_config.sh

if [[ "$?" -ne 0 ]]; then
	exit 1
fi

# ensure cache exists
mkdir -p .cache

echo "Mounting containers specified in mounts.txt using blobfuse2..."
TO_MOUNT=(
	"nssp-etl"
	"nssp-archival-vintages"
	"prod-param-estimates"
	"pyrenew-hew-prod-output"
	"pyrenew-test-output"
	"nwss-vintages"
	"pyrenew-hew-config"
)

for dir in "${TO_MOUNT[@]}"; do
	echo "Mounting" $dir
	mkdir -p /mnt/$dir
	blobfuse2 mount --container-name $dir /mnt/$dir --allow-other
done

echo ""
echo "Creating symlinks in $(pwd)..."

ln -s "/mnt/prod-param-estimates" "./params"
ln -s "/mnt/pyrenew-hew-prod-output" "./output"
ln -s "/mnt/pyrenew-test-output" "./test-output"
ln -s "/mnt/nwss-vintages" "./nwss-vintages"
ln -s "/mnt/pyrenew-hew-config" "./config"
ln -s "/mnt/nssp-etl" "./nssp-etl"
ln -s "/mnt/nssp-archival-vintages" "./nssp-archival-vintages"

echo "Setting environment variables for the Azure Command Center..."
export NSSP_ETL_PATH="$(pwd)/nssp-etl"
export PYRENEW_HEW_PROD_OUTPUT_PATH="$(pwd)/output"
export NWSS_VINTAGES_PATH="$(pwd)/nwss-vintages"

echo "Done."
