REMOTEDIR=/tmp/tracker_output
LOCDIR=/tmp/synced_tracker_output
mkdir -p $LOCDIR
rsync -rvz --ignore-existing -e 'ssh -p 58022' --progress  aljosa@atcremers73.vision.in.tum.de:$REMOTEDIR $LOCDIR

