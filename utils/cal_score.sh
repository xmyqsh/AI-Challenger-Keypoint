#sh cal_score.sh checkpoints/AI/AI_hg8/pred_post_6.h5 data/AI/valid_images.txt data/AI/keypoint_predictions.json data/AI/keypoint_predictions.json data/AI/annotations/validation.json

python utils/prediction_json_gen.py --pred_path $1 \
                                    --imgset_path $2 \
                                    --output_path $3

python utils/keypoint_eval.py --submit $4 \
                              --ref $5 \
                              --retLevel $6
