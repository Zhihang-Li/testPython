
import json
import os
import skimage.io as io
#matplotlib.use('agg')
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import unicodedata

plt.switch_backend('agg')

def showResults(img, image_name, save_path, results, txt, save_fig=False):
    # img:
    # note: [x_min,y_min,w,h]   not [x_min, y_min, x_max, y_max]
    plt.clf()
    plt.imshow(img)
    plt.axis('off')
    ax = plt.gca()

    # num_classes = len(labelmap.item) - 1
    colors = plt.cm.hsv(np.linspace(0, 1, 10)).tolist()
    color = colors[4]

    for i in range(0, results.shape[0]):
        # score = results[i, -2]
        score = results
        # if threshold and score < threshold:
        #     continue

        # label = int(results[i, -1])
        # name = get_labelname(labelmap, label)[0]
        # color = colors[label % num_classes]

        xmin = int(round(results[i, 0]))
        ymin = int(round(results[i, 1]))
        xmax = int(round(results[i, 2]))
        ymax = int(round(results[i, 3]))
        # coords = (xmin, ymin), xmax - xmin, ymax - ymin
        coords = (xmin, ymin), xmax, ymax
        ax.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=1))
        # display_text = '%s: %.2f' % (name, score)
        ax.text(xmin, ymin, txt[i], bbox={'facecolor':color, 'alpha':0.5},fontsize=3)
    if save_fig:
        plt.savefig(os.path.join(save_path, image_name[:-4] + '_dets.jpg'), bbox_inches="tight")
        print('Saved: ' + image_name[:-4] + '_dets.jpg')
    plt.show()


is_plot_fbox = True
is_plot_vbox = True
is_plot_hbox = True

json_path = 'annotation_train.odgt'
root_image = 'Images/'
#vbox_save_path = 'vbox_plot_image/'
#fbox_save_path = 'fbox_plot_image/'
box_save_path = 'vbox_plot_image/'
if not os.path.exists(box_save_path):
    os.mkdir(box_save_path)


with open(json_path, 'r') as file:
    for line in file.readlines():
        # read json to dict
        line = json.loads(line.strip())
        # ID: str
        ID = line['ID']
        ID = unicodedata.normalize('NFKD', ID).encode('ascii','ignore')
        img_name = ID + '.jpg'
        img_path = os.path.join(root_image, img_name)
        img = io.imread(img_path)
        # gtboxes: list, every person
        gtboxes = line['gtboxes']
        fbox_list = []
        vbox_list = []
        hbox_list = []
        txt = []
        for gtbox in gtboxes:
            	# gtbox: dict, [u'fbox', u'extra', u'head_attr', u'vbox', u'tag', u'hbox']
            	# extra and headattr
            extra  = gtbox['extra']
            head_attr = gtbox['head_attr']

                # list
            if (('ignore' in extra) and extra['ignore'] == 0) or (('ignore' in head_attr) and head_attr['ignore'] == 0):
                fbox_list.append(np.asarray(gtbox['fbox']))
                fbox_tensor = np.row_stack(fbox_list)
                vbox_list.append(np.asarray(gtbox['vbox']))
                vbox_tensor = np.row_stack(vbox_list)
                hbox_list.append(np.asarray(gtbox['hbox']))
                hbox_tensor = np.row_stack(hbox_list)
                tmp = ''
                if 'occ' in extra:
                    tmp = tmp + str(extra['occ'])
                txt.append(tmp)
        showResults(img, img_name, box_save_path, vbox_tensor, txt, save_fig=True)


