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
mkdir -p ./mounts
echo "Creating symlinks in $(pwd)/mounts..."

# Create symlinks only if they do not already exist, and inform the user
if [[ -L "./mounts/params" ]]; then
	echo "Symlink './mounts/params' already exists, skipping."
else
	ln -s "/mnt/prod-param-estimates" "./mounts/params"
	echo "Created symlink './mounts/params' -> '/mnt/prod-param-estimates'"
fi

if [[ -L "./mounts/output" ]]; then
	echo "Symlink './mounts/output' already exists, skipping."
else
	ln -s "/mnt/pyrenew-hew-prod-output" "./mounts/output"
	echo "Created symlink './mounts/output' -> '/mnt/pyrenew-hew-prod-output'"
fi

if [[ -L "./mounts/test-output" ]]; then
	echo "Symlink './mounts/test-output' already exists, skipping."
else
	ln -s "/mnt/pyrenew-test-output" "./mounts/test-output"
	echo "Created symlink './mounts/test-output' -> '/mnt/pyrenew-test-output'"
fi

if [[ -L "./mounts/nwss-vintages" ]]; then
	echo "Symlink './mounts/nwss-vintages' already exists, skipping."
else
	ln -s "/mnt/nwss-vintages" "./mounts/nwss-vintages"
	echo "Created symlink './mounts/nwss-vintages' -> '/mnt/nwss-vintages'"
fi

if [[ -L "./mounts/config" ]]; then
	echo "Symlink './mounts/config' already exists, skipping."
else
	ln -s "/mnt/pyrenew-hew-config" "./mounts/config"
	echo "Created symlink './mounts/config' -> '/mnt/pyrenew-hew-config'"
fi

if [[ -L "./mounts/nssp-etl" ]]; then
	echo "Symlink './mounts/nssp-etl' already exists, skipping."
else
	ln -s "/mnt/nssp-etl" "./mounts/nssp-etl"
	echo "Created symlink './mounts/nssp-etl' -> '/mnt/nssp-etl'"
fi

if [[ -L "./mounts/nssp-archival-vintages" ]]; then
	echo "Symlink './mounts/nssp-archival-vintages' already exists, skipping."
else
	ln -s "/mnt/nssp-archival-vintages" "./mounts/nssp-archival-vintages"
	echo "Created symlink './mounts/nssp-archival-vintages' -> '/mnt/nssp-archival-vintages'"
fi

echo "Setting environment variables for the Azure Command Center..."
export NSSP_ETL_PATH="$(pwd)/mounts/nssp-etl"
export PYRENEW_HEW_PROD_OUTPUT_PATH="$(pwd)/mounts/output"
export NWSS_VINTAGES_PATH="$(pwd)/mounts/nwss-vintages"

echo "Done."
