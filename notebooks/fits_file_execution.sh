#!/bin/bash
# filepath: /home/annalena/rubix/notebooks/fits_file_execution.sh

# Set the telescope name parameter
TELESCOPE_NAME="MUSE_ultraWFM"

# Loop through the galaxy IDs
for GALAXY_ID in g1.08e11 g1.37e11 g1.52e11 g1.59e11 g2.41e11 g2.42e11 g2.57e11 g3.06e11 g3.49e11 g3.61e11 g4.90e11 g5.02e11 g5.31e11 g5.38e11 g5.46e11 g5.55e11 g6.96e11 g7.55e11 g7.66e11 g8.13e11 g8.26e11 g1.12e12 g1.77e12 g1.92e12 g2.79e12 ; do
    echo "Processing galaxy_id: $GALAXY_ID"

    # Remove the rubix_galaxy.h5 file if it exists for this galaxy
    rm -f output/rubix_galaxy.h5

    # Loop through the pipeline names and execute the notebook for each
    for PIPELINE_NAME in calc_ifu ; do
        echo "Running Papermill for pipeline: $PIPELINE_NAME with galaxy_id: $GALAXY_ID at telescope: $TELESCOPE_NAME"
        papermill fits_file_nihao_script.ipynb \
            fits_file_nihao_script_executed.ipynb \
            -p galaxy_id "$GALAXY_ID" \
            -p pipeline_name "$PIPELINE_NAME" \
            -p telescope_name "$TELESCOPE_NAME"
    done
done
