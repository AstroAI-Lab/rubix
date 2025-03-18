#!/bin/bash
# filepath: /home/annalena/rubix/notebooks/fits_file_execution.sh

# Remove the rubix_galaxy.h5 file if it exists
rm -f output/rubix_galaxy.h5

# Set the parameter values
GALAXY_ID="g8.26e11"
PIPELINE_NAME="calc_dusty_ifu"
TELESCOPE_NAME="MUSE_test"

# Execute the notebook using Papermill, passing the parameters
papermill fits_file_nihao_script.ipynb fits_file_nihao_script_executed.ipynb \
    -p galaxy_id "$GALAXY_ID" \
    -p pipeline_name "$PIPELINE_NAME" \
    -p telescope_name "$TELESCOPE_NAME"
