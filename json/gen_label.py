
import json
import os
import skimage.io as io
#matplotlib.use('agg')
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import unicodedata

def xywh2xy(bbox):
    # bbox: list, [x, y, w, h]
    # return: list [x1,y1, x2,y2 ..., 1]
    new_bbox = []
    new_bbox.append(bbox[0])
    new_bbox.append(bbox[1])
    new_bbox.append(bbox[0] + bbox[2])
    new_bbox.append(bbox[1])
    new_bbox.append(bbox[0] + bbox[2])
    new_bbox.append(bbox[1] + bbox[3])
    new_bbox.append(bbox[0])
    new_bbox.append(bbox[1] + bbox[3])

    # score
    new_bbox.append(1.0)
    return new_bbox

json_path = 'annotation_train.odgt'

# file path
fbox_save_path = 'fbox_label.txt'
vbox_save_path = 'vbox_label.txt'
hbox_save_path = 'hbox_label.txt'

# open file 
fbox_f = open(fbox_save_path, 'wa')
vbox_f = open(vbox_save_path, 'wa')
hbox_f = open(hbox_save_path, 'wa')

# if not os.path.exists(box_save_path):
#     os.mkdir(box_save_path)


with open(json_path, 'r') as file:
    for line in file.readlines():
        # read json to dict
        line = json.loads(line.strip())
        # ID: str
        ID = line['ID']
        ID = unicodedata.normalize('NFKD', ID).encode('ascii','ignore')
        img_name = ID + '.jpg'
        # img_path = os.path.join(root_image, img_name)
        # img = io.imread(img_path)
        # gtboxes: list, every person
        gtboxes = line['gtboxes']
        fbox_list = []
        vbox_list = []
        hbox_list = []
        txt = []
        i = 0
        for gtbox in gtboxes:
            # gtbox: dict, [u'fbox', u'extra', u'head_attr', u'vbox', u'tag', u'hbox']
            # extra and headattr
            extra  = gtbox['extra']
            head_attr = gtbox['head_attr']

            # list
            if (('ignore' in extra) and extra['ignore'] == 0) or (('ignore' in head_attr) and head_attr['ignore'] == 0):
                i = i + 1
                fbox = xywh2xy(gtbox['fbox'])
                fbox_list.append(fbox)
                vbox = xywh2xy(gtbox['vbox'])
                vbox_list.append(vbox)
                hbox = xywh2xy(gtbox['hbox'])
                hbox_list.append(hbox)
        # write img_name, 
        fbox_f.write(ID + '\n')
        vbox_f.write(ID + '\n')
        hbox_f.write(ID + '\n')
        # write image num
        fbox_f.write(str(i) + '\n')
        vbox_f.write(str(i) + '\n')
        hbox_f.write(str(i) + '\n')
        # write bbox and score
        for fbox in fbox_list:
            fbox = " ".join(str(x) for x in fbox)
            fbox_f.write(fbox + '\n')

        for vbox in vbox_list:
            vbox = " ".join(str(x) for x in vbox)
            vbox_f.write(vbox + '\n')

        for hbox in hbox_list:
            hbox = " ".join(str(x) for x in hbox)
            hbox_f.write(hbox + '\n')

        # fbox_str = fbox_list
        # fbox_f.write(fbox_str)

        # vbox_str = vbox_list
        # vbox_f.write(vbox_str)

        # hbox_str = hbox_list
        # hbox_f.write(hbox_str)

fbox_f.close()
vbox_f.close()
hbox_f.close()
