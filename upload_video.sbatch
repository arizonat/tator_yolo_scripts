#!/usr/bin/env bash
#SBATCH --job-name=video2tator
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=2GB
#SBATCH --time=6:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hsosik@whoi.edu
#SBATCH --partition=compute
#SBATCH --output=slogs/video2tator/%j.%x.txt

# LOGGING JOB DETAILS #
echo "Job ID: $SLURM_JOB_ID, JobName: $SLURM_JOB_NAME"
hostname; pwd; date

# SETTING UP ENVIRONMENT #
module load ffmpeg
module load gcc/8.5.0
module load python3/3.9.10
source venv/bin/activate
umask 000  # newly created files have all permissions ok. 
PATH="$(realpath bin):$PATH"  # add bin/mp4dump to PATH
echo "Environment... loaded"

set -eux  # exit on error, including unset vars

# Get target dir to from list and slurm-array-index to convert to video
VIDEO_FILE="$1"

## TRANSCODING AND UPLOADING TO TATOR ##

echo "TATOR-PY..."
TOKEN=$(cat tator_token.txt)  # hpc-user access token
HOST=https://tator.whoi.edu
PROJ_ID=1   # ISIIS
MEDIA_ID=1  # shadowgraph video
FOLDER="$(basename $VIDEO_FILE | cut -f2 -d'_')"

time python3 -m tator.transcode "$VIDEO_FILE" \
    --host $HOST \
    --token $TOKEN \
    --project $PROJ_ID \
    --type $MEDIA_ID \
    --section "$FOLDER" \
    --work_dir "tator_transcode_workspace" 

echo 
TZ=UTC0 printf '%(%H:%M:%S)T\n' $SECONDS


