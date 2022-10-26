#!/usr/bin/env bash
#SBATCH --job-name=tiff2video
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=2GB
#SBATCH --time=6:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hsosik@whoi.edu
#SBATCH --partition=compute
#SBATCH --output=slogs/tiff2video/%j.%x.txt

# LOGGING JOB DETAILS #
echo "Job ID: $SLURM_JOB_ID, JobName: $SLURM_JOB_NAME"
hostname; pwd; date

# SETTING UP ENVIRONMENT #
module load ffmpeg
module load gcc/8.5.0
module load imagemagick 
module load python3/3.9.10
source venv/bin/activate
umask 000  # newly created files have all permissions ok. 
echo "Environment... loaded"

set -eu  # exit on error, including unset vars

#SRC=/vortexfs1/share/otz-data/AR43_Stingray/data
#SRC=/vortexfs1/share/nes-lter/Stingray/data

DST=tiff2video_output

# Video directory is first argument
VIDEO_DIR=$1
VIDEO=$(basename $VIDEO_DIR)

# Get filename format of the directory
FILEFMT=$(ls -p "$VIDEO_DIR" | grep -v / | head -n1)
FILEFMT=${FILEFMT%_*}

# Determine if there are leading zero's or not
test -f "$VIDEO_DIR/${FILEFMT}_1.tiff" && LEADINGZERO=1
test -f "$VIDEO_DIR/${FILEFMT}_0001.tiff" && LEADINGZERO=4

# Display action
echo "$VIDEO_DIR/${FILEFMT}_%0${LEADINGZERO}d.tiff --> $DST/$VIDEO.mp4" 
echo


## FRAMES TO VIDEO ##

echo "FFMPEG..."
# creating title slide
IMG1="$VIDEO_DIR/${FILEFMT}_$(printf "%0${LEADINGZERO}d" 1).tiff"  # extant first image
IMG0="$VIDEO_DIR/${FILEFMT}_$(printf "%0${LEADINGZERO}d" 0).tiff"  # new title slide image to create

img_width=`identify -format %w $IMG1`
img_height=`identify -format %h $IMG1`
boxsize="$((9*img_width/10))x$((2*img_height/10))"
# boxwidth 90% of frame, boxheight 20% of frame
# create centered caption overlay box, save to IMG0
magick -background '#0008' -fill white -gravity center -size $boxsize \
      -font DejaVu-Sans -pointsize 100 caption:"$VIDEO" \
      $IMG1 +swap -gravity center -composite  $IMG0


# creating bonus final slide
IMGm=$(ls $VIDEO_DIR/${FILEFMT}_*.tiff | sort -V | tail -n 1)
n=$(ls $VIDEO_DIR/${FILEFMT}_*.tiff | wc -l)  # works because it includes the temporary IMG0
IMGn=$VIDEO_DIR/${FILEFMT}_$(printf "%0${LEADINGZERO}d" $n).tiff
if [ ! -f $IMGn ];then
    cp $IMGm $IMGn
fi # else there is one or more missing files and that's bad. don't overwite them!

# ensure presense of target directory
mkdir -p "$DST"

# creating video file
time ffmpeg -framerate 15 -i "$VIDEO_DIR/${FILEFMT}_%0${LEADINGZERO}d.tiff" -c:v libx265 -crf 23 -pix_fmt yuv420p "$DST/$VIDEO.mp4" 

# cleanup: removing generated images and frame files list
rm $IMG0 $IMGn 




echo 
TZ=UTC0 printf '%(%H:%M:%S)T\n' $SECONDS

sleep 10


