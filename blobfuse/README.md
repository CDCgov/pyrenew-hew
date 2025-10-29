# Pyrenew Blobfuse Configuration

This directory serves as a project-specific fork of the [cfa-blobfuse-tutuorial](https://github.com/cdcent).
This directory will mount pyrenew-hew blobs to `/mnt` and then symlink to a directory you specify (or the current directory if you don't supply an argument).

To run, make sure you're in the top level as your working directory (`pyrenew-hew`, and not `pyrenew-hew/blobfuse`).
1. Run `sudo chmod +x ./blobfuse/mount.sh`.
2. Run `sudo ./blobfuse/mount.sh <name_of_dir_to_symlink_blobs`.
3. Check to make sure `/mnt` has pyrenew blobs mounted and that symlinks have been created in your working directory (`pyrenew-hew/`).