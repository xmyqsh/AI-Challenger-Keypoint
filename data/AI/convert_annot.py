import json
import h5py
import numpy as np
import sys

#keys = ['index','person','imgname','center','scale','part','visible','normalize','torsoangle','multi','istrain']
keys = ['part', 'center', 'scale', 'visible']
annot = {k:[] for k in keys}

imageset = 'train' # validation

imagesSetsPaths = './' + imageset + '_images.txt'
'''
with open(imagesSetsPaths) as fid:
    return
'''

images_fid = open(imagesSetsPaths, 'w')

annotPath = './annotations/' + imageset + '.json'
with open(annotPath, 'r') as fid:
    labels = json.load(fid)

nimages = len(labels)

for idx in xrange(nimages):
    print "\r",idx,
    sys.stdout.flush()

    image_annot = labels[idx]
    imagePath = '/'.join([imageset, image_annot['image_id'] + '.jpg'])
    image_humans_annot = image_annot['human_annotations']
    image_keypoints_annot = image_annot['keypoint_annotations']
    for person in image_humans_annot.keys():
        images_fid.write(imagePath)
        images_fid.write('\n')

        human_annot = image_humans_annot[person]
        x1, y1, x2, y2 = human_annot
        c = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float)
        s = np.array([(x2 - x1) / 200.0, (y2 - y1) / 200.0], dtype=np.float)
        annot['center'].append(c)
        annot['scale'].append(s)

        keypoint_annot = image_keypoints_annot[person]
        coords = np.zeros((14,2), dtype=np.float)
        vis = np.zeros(14, dtype=np.float)
        for index in xrange(0, 42, 3):
            coords[index / 3] = np.array([keypoint_annot[index], keypoint_annot[index + 1]], dtype=np.float)
            vis[index / 3] = keypoint_annot[index + 2]
        annot['part'].append(coords)
        annot['visible'].append(vis)
        '''
        if not c[0] == -1:
            # Add info to annotation list
            annot['index'] += [idx]
            annot['person'] += [person]
            imgname = np.zeros(16)
            refname = str(imgnameRef[idx][0][0][0][0])
            for i in range(len(refname)): imgname[i] = ord(refname[i])
            annot['imgname'] += [imgname]
            annot['center'] += [c]
            annot['scale'] += [s]

            if mpii.istrain(idx) == True:
                # Part annotations and visibility
                coords = np.zeros((16,2))
                vis = np.zeros(16)
                for part in xrange(16):
                   coords[part],vis[part] = mpii.partinfo(idx,person,part)
                annot['part'] += [coords]
                annot['visible'] += [vis]
                annot['normalize'] += [mpii.normalization(idx,person)]
                annot['torsoangle'] += [mpii.torsoangle(idx,person)]
                annot['istrain'] += [1]
            else:
                annot['part'] += [-np.ones((16,2))]
                annot['visible'] += [np.zeros(16)]
                annot['normalize'] += [1]
                annot['torsoangle'] += [0]
                if trainRef[idx] == 0:  # Test image
                    annot['istrain'] += [0]
                else:   # Training image (something missing in annot)
                    annot['istrain'] += [2]
        '''

print ""


images_fid.close()

with h5py.File(imageset + '.h5','w') as f:
    f.attrs['name'] = 'AI'
    for k in keys:
        f[k] = np.array(annot[k])
