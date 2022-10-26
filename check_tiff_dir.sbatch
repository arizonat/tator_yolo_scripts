#!/usr/bin/env bash
#SBATCH --job-name=tiffcheck
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2GB
#SBATCH --time=6:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hsosik@whoi.edu
#SBATCH --partition=compute
#SBATCH --output=slogs/tiffcheck/%j.%x.txt

# LOGGING JOB DETAILS #
echo "Job ID: $SLURM_JOB_ID, JobName: $SLURM_JOB_NAME"
hostname; pwd; date

VIDEO_DIR="$1"

VIDEO=$(basename "$VIDEO_DIR")

FILEFMT=$(ls "$VIDEO_DIR" | head -n1)
FILEFMT=${FILEFMT%_*}

echo "$VIDEO_DIR/${FILEFMT}_%d.tiff"
echo "Searching for missing frames..."
echo

END=$(ls $VIDEO_DIR | wc -l)

test -f "$VIDEO_DIR/${FILEFMT}_1.tiff" && LEADINGZERO=1
test -f "$VIDEO_DIR/${FILEFMT}_01.tiff" && LEADINGZERO=2
test -f "$VIDEO_DIR/${FILEFMT}_001.tiff" && LEADINGZERO=3
test -f "$VIDEO_DIR/${FILEFMT}_0001.tiff" && LEADINGZERO=4
test -f "$VIDEO_DIR/${FILEFMT}_00001.tiff" && LEADINGZERO=5
test -f "$VIDEO_DIR/${FILEFMT}_000001.tiff" && LEADINGZERO=6

echo "LEADINGZERO = ${LEADINGZERO}"

OUTFILE="tiffcheck_output/$VIDEO.txt"
rm $OUTFILE  # start clean
mkdir -p $(dirname $OUTFILE)

for (( i=1; i<=$END; i++ ))
do
    if [ ! -f "$VIDEO_DIR/${FILEFMT}_$(printf "%0${LEADINGZERO}d" $i).tiff" ]; then

        echo "$VIDEO/${FILEFMT}_$(printf "%0${LEADINGZERO}d" $i).tiff" | tee -a $OUTFILE

    fi
done

echo
echo "DONE"

