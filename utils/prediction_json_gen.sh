#sh prediction_json_gen.sh checkpoints/AI/AI_hg8/pred_post_6.h5 data/AI/valid_images.txt data/AI/keypoint_predictions.json

python utils/prediction_json_gen.py --pred_path $1 \
                                    --imgset_path $2 \
                                    --output_path $3
