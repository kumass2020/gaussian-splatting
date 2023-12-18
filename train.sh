echo "train"
python train.py -s download/tandt_db/tandt/train --eval
echo "truck"
python train.py -s download/tandt_db/tandt/truck --eval