#!/bin/bash

### Define the array of nside values
nsides=(2048 4096 8192)

### Loop through each nside value and submit the job
for nside in "${nsides[@]}"; do
        sbatch --export=ALL,NSIDE=$nside run_create_catalog_background.slurm
done
