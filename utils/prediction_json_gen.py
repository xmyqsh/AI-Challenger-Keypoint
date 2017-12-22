import json
import h5py
import argparse


def prediction_gen(pred_path, imgset_path, output_path):
    #pred_path = 'checkpoints/AI/AI_hg8/pred_post_8.h5'
    pred = h5py.File(pred_path, 'r')
    preds = pred['preds']

    #imgset_path = 'data/AI/valid_images.txt'
    #imgset_path = 'data/AI/det_validation_images.txt'
    fid = open(imgset_path, 'r')

    img2pred = {}

    for i in xrange(preds.shape[0]):
        imgId = fid.readline().strip().split('.')[0].split('/')[1]
        #print imgId

        if imgId not in img2pred:
            img2pred[imgId] = []
        human = []
        for j in xrange(preds.shape[1]):
            human.append(int(preds[i][j][0]))
            human.append(int(preds[i][j][1]))
            human.append(1)
        img2pred[imgId].append(human)

    fid.close()
    pred.close()

    print len(img2pred.keys())

    jsondict = []
    for key, value in img2pred.items():
        entry = {}
        entry['image_id'] = key
        entry['keypoint_annotations'] = {}
        for idx, human in enumerate(value):
            entry['keypoint_annotations']['human' + str(idx + 1)] = human
        jsondict.append(entry)

    '''
    print jsondict[0].keys()
    for key, value in jsondict[0].items():
        print key, value
    '''

    #outputPath = 'keypoint_predictions.json'
    with open(output_path, 'w') as fid:
        encoded_jsondict = json.dumps(jsondict)
        fid.writelines(encoded_jsondict)


def main():
    """The evaluator."""

    # Arguments parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_path', help='prediction h5 file', type=str,
                        default='checkpoints/AI/AI_hg8/pred_post_8.h5')
    parser.add_argument('--imgset_path', help='imageset txt file', type=str,
                        default='data/AI/valid_images.txt')
    parser.add_argument('--output_path', help='prediction json file', type=str,
                        default='keypoint_predictions.json')

    args = parser.parse_args()

    prediction_gen(pred_path=args.pred_path, imgset_path=args.imgset_path, output_path=args.output_path)

if __name__ == "__main__":
    main()
