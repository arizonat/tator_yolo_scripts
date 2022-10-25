#!/usr/bin/env bash
#SBATCH --job-name=yolo2tator
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4GB
#SBATCH --time=3:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hsosik@whoi.edu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=slogs/yolo2tator/%j.%x.txt

set -x

# LOGGING JOB DETAILS #
echo "Job ID: $SLURM_JOB_ID, JobName: $SLURM_JOB_NAME"
hostname; pwd; date

module load cuda10.1/toolkit cuda10.1/blas cuda10.1/cudnn/8.0.2 cuda91/fft
module load python3/3.9.10
source venv/bin/activate
echo "Environment... loaded"

set -eu

#MEDIA=$1
#MODEL=$2
#CLASSLIST=$3
#time python run_yolo_on_video_and_upload_rois.py "$MEDIA" "$MODEL" --device=0 --classlist "$CLASSLIST" --conf-thres 0.1 --img-size 2330

TOKEN=$(cat tator_token.txt)  # hpc-user access token

time python3 run_yolo_model_on_video_and_upload_rois.py "$@" --device=0 --token $TOKEN


