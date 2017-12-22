#sh keypoint_eval.sh data/AI/keypoint_predictions.json data/AI/annotations/validation.json 0/1

python utils/keypoint_eval.py --submit $1 \
                              --ref $2 \
                              --retLevel $3
