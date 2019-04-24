
import cv2
import os

def correct_bbox(bbox_list, H, W):
	# bbox_list: x, y, w, h
	x = int(bbox_list[0])
	y = int(bbox_list[1])
	w = int(bbox_list[2])
	h = int(bbox_list[3])
	x_max = x + w
	y_max = y + h

	if x_max > W:
		x_max = W
		print('---correct---')
	if y_max > H:
		y_max = H

	new_bbox = [x, y, x_max-x, y_max-y]
	return new_bbox


bbox_txt_path = '/root/dukang/data/lizhihang/human_detection/WIDE_pedestrain/WIDE_pedestrain/pedestrian_detection_trainval/train_annotations.txt'
img_base_path = '/root/dukang/data/lizhihang/human_detection/WIDE_pedestrain/WIDE_pedestrain/pedestrian_detection_trainval/train'

save_bbox_path = '/root/dukang/data/lizhihang/human_detection/WIDE_pedestrain/WIDE_pedestrain/pedestrian_detection_trainval/train_annotations_corr.txt'
write_file = open(save_bbox_path, 'w')


with open(bbox_txt_path, 'r') as file:
	for line in file.readlines():
		# save write line
		save_line = ''

		iterms = line.split(' ')
		img_name = iterms[0]

		save_line = save_line + img_name
		# read image to get w and h
		img_path = os.path.join(img_base_path, img_name)
		if not os.path.exists(img_path):
			print(img_path)

		img = cv2.imread(os.path.join(img_base_path, img_name))
		
		H = img.shape[0]
		W = img.shape[1]

		num_bbox = (len(iterms)-1) / 5
		for i in range(num_bbox):
			label = iterms[i*5+1]
			bbox_list = iterms[i*5+2: i*5+6]
			new_box = correct_bbox(bbox_list, H, W)
			save_line = save_line + ' ' + label
			for j in new_box:
				save_line = save_line + ' ' + str(j)
		save_line = save_line + '\n'
		write_file.write(save_line)

write_file.close()


