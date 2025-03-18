#!/bin/bash
# filepath: /home/annalena/rubix/notebooks/fits_file_execution.sh

# Set the telescope name parameter
TELESCOPE_NAME="MUSE_test"

# Loop through the galaxy IDs
for GALAXY_ID in g7.55e11 g7.66e11 g8.06e11 g8.13e11 g8.26e11 g8.28e11 g1.12e12 g1.77e12 g1.92e12 g2.79e12; do
    echo "Processing galaxy_id: $GALAXY_ID"

    # Remove the rubix_galaxy.h5 file if it exists for this galaxy
    rm -f output/rubix_galaxy.h5

    # Loop through the pipeline names and execute the notebook for each
    for PIPELINE_NAME in calc_ifu calc_dusty_ifu; do
        echo "Running Papermill for pipeline: $PIPELINE_NAME with galaxy_id: $GALAXY_ID"
        papermill fits_file_nihao_script.ipynb \
            fits_file_nihao_script_executed.ipynb \
            -p galaxy_id "$GALAXY_ID" \
            -p pipeline_name "$PIPELINE_NAME" \
            -p telescope_name "$TELESCOPE_NAME"
    done
done
