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

# Create symlinks only if they do not already exist, and inform the user
if [[ -L "./params" ]]; then
	echo "Symlink './params' already exists, skipping."
else
	ln -s "/mnt/prod-param-estimates" "./params"
	echo "Created symlink './params' -> '/mnt/prod-param-estimates'"
fi

if [[ -L "./output" ]]; then
	echo "Symlink './output' already exists, skipping."
else
	ln -s "/mnt/pyrenew-hew-prod-output" "./output"
	echo "Created symlink './output' -> '/mnt/pyrenew-hew-prod-output'"
fi

if [[ -L "./test-output" ]]; then
	echo "Symlink './test-output' already exists, skipping."
else
	ln -s "/mnt/pyrenew-test-output" "./test-output"
	echo "Created symlink './test-output' -> '/mnt/pyrenew-test-output'"
fi

if [[ -L "./nwss-vintages" ]]; then
	echo "Symlink './nwss-vintages' already exists, skipping."
else
	ln -s "/mnt/nwss-vintages" "./nwss-vintages"
	echo "Created symlink './nwss-vintages' -> '/mnt/nwss-vintages'"
fi

if [[ -L "./config" ]]; then
	echo "Symlink './config' already exists, skipping."
else
	ln -s "/mnt/pyrenew-hew-config" "./config"
	echo "Created symlink './config' -> '/mnt/pyrenew-hew-config'"
fi

if [[ -L "./nssp-etl" ]]; then
	echo "Symlink './nssp-etl' already exists, skipping."
else
	ln -s "/mnt/nssp-etl" "./nssp-etl"
	echo "Created symlink './nssp-etl' -> '/mnt/nssp-etl'"
fi

if [[ -L "./nssp-archival-vintages" ]]; then
	echo "Symlink './nssp-archival-vintages' already exists, skipping."
else
	ln -s "/mnt/nssp-archival-vintages" "./nssp-archival-vintages"
	echo "Created symlink './nssp-archival-vintages' -> '/mnt/nssp-archival-vintages'"
fi

echo "Setting environment variables for the Azure Command Center..."
export NSSP_ETL_PATH="$(pwd)/nssp-etl"
export PYRENEW_HEW_PROD_OUTPUT_PATH="$(pwd)/output"
export NWSS_VINTAGES_PATH="$(pwd)/nwss-vintages"

echo "Done."
