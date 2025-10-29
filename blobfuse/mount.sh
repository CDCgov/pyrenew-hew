#!/bin/bash

# ensure logged in via Azure CLI.
./blobfuse/verifylogin.sh

# pull azure configuration files
./blobfuse/pull_config.sh

if [[ "$?" -ne 0 ]]
then
    exit 1
fi

# ensure cache exists
mkdir -p .cache

echo "Mounting containers specified in mounts.txt using blobfuse2..."

TO_MOUNT=$(<mounts.txt)

for dir in $TO_MOUNT
do
    echo "Mounting" $dir
    mkdir -p /mnt/$dir
    blobfuse2 mount --container-name $dir /mnt/$dir --allow-other
done
sym_dir="${1:=.}"
echo ""
echo "Creating symlinks in $sym_dir..."
ln -s "/mnt/nssp-etl" "$sym_dir/nssp-etl/"
ln -s "/mnt/prod-param-estimates" "$sym_dir/params"
ln -s "/mnt/pyrenew-hew-prod-output" "$sym_dir/output"
ln -s "/mnt/pyrenew-test-output" "$sym_dir/test-output"
ln -s "/mnt/nssp-archival-vintages" "$sym_dir/nssp-archival-vintages/"
ln -s "/mnt/nwss-vintages" "$sym_dir/nwss-vintages"
ln -s "/mnt/pyrenew-hew-config" "$sym_dir/config"

echo "Done."