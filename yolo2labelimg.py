import sys
# darknet_dir = '/home/srv2019/Documents/GA/darknet'
# color_dir = '/home/srv2019/Documents/phenology_distribution'
# sys.path.append(darknet_dir)
# sys.path.append(color_dir)
# from draw_distribution import colors
# from darknet import performDetect
import os, glob
import cv2
import numpy as np
from skimage.measure import label, regionprops
import xml.etree.cElementTree as ET
from matplotlib import pyplot as plt

showImage_flag = False


########################YOLO configuration##########################################
darknet_dir = '/home/srv2019/Documents/GA/darknet'
sys.path.append(darknet_dir)
from darknet import performDetect
configPath = "/home/srv2019/Documents/GA/darknet/phenology/yolov4.cfg"
weightPath = "/home/srv2019/Documents/GA/darknet/phenology/backup/yolov4_600.weights"
metaPath = "/home/srv2019/Documents/GA/darknet/phenology/green.data"
classnames = '/home/srv2019/Documents/GA/darknet/phenology/classes.names'
####################################################################################

with open(classnames) as f:
    classes = f.read().splitlines()
classes = np.array(classes) 
result_dir = 'yolov4_results'

#########################Quad configuration########################################
quad_root = '../phenology_distribution/all_quads'
quad_dirs = ['627_16B_GPS', '642_16A_GPS']
###################################################################################

def detections2xml(detections, xmlfile, image, minr,minc,maxr,maxc):

	detections = np.array(detections)
	labels = detections[:,0]
	scores = detections[:,1]
	bboxes = np.array([np.array(i) for i in detections[:,2]])


	root = ET.Element("annotation")
	folder = ET.SubElement(root, "folder").text = 'quads_images'
	filename = ET.SubElement(root, "filename").text = '{}'.format(os.path.basename(xmlfile)[:-3]+'jpg')
	path = ET.SubElement(root, "path").text = '{}'.format(xmlfile[:-3]+'jpg')
	source = ET.SubElement(root, "source")
	database = ET.SubElement(source, "database").text = 'Unknown'
	size = ET.SubElement(root, "size")
	width = ET.SubElement(size, "width").text = '{}'.format(image.shape[1])
	height = ET.SubElement(size, "height").text = '{}'.format(image.shape[0])
	depth = ET.SubElement(size, "depth").text = '{}'.format(image.shape[2])
	segmented = ET.SubElement(root, "segmented").text= '0'

	for label, bbox in zip (labels, bboxes):

		top_left = (int(bbox[0]-bbox[2]//2+minc), int(bbox[1]-bbox[3]//2+minr))
		bot_right = (int(bbox[0]+bbox[2]//2+minc), int(bbox[1]+bbox[3]//2+minr))
		
		color_index = np.where(classes == label)[0][0]
		# cv2.putText(image, label, top_left, cv2.FONT_HERSHEY_SIMPLEX , 0.5,  colors[color_index], 1, cv2.LINE_AA)
		# cv2.rectangle(image, top_left, bot_right, colors[color_index], 2)

		object_root = ET.SubElement(root, "object")
		name = ET.SubElement(object_root, "name").text='{}'.format(label)
		pose = ET.SubElement(object_root, "pose").text='Unspecified'
		truncated = ET.SubElement(object_root, "truncated").text='0'
		difficult = ET.SubElement(object_root, "difficult").text='0'
		bndbox = ET.SubElement(object_root, "bndbox")
		xmin = ET.SubElement(bndbox, "xmin").text='{}'.format(top_left[0])
		ymin = ET.SubElement(bndbox, "ymin").text='{}'.format(top_left[1])
		xmax = ET.SubElement(bndbox, "xmax").text='{}'.format(bot_right[0])
		ymax = ET.SubElement(bndbox, "ymax").text='{}'.format(bot_right[1])

	tree = ET.ElementTree(root)
	tree.write(xmlfile)
	# cv2.imwrite('test.png', image)


for quad_dir in quad_dirs:

	images_path = glob.glob(os.path.join(quad_root, quad_dir, 'quads_images', '*.jpg'))
	images_path = sorted(images_path, key=lambda x:int(os.path.basename(x).split('_')[0][3:]))
	
	masks_path = glob.glob(os.path.join(quad_root, quad_dir, 'mask_quads', '*.jpg'))
	masks_path = sorted(masks_path, key=lambda x:int(os.path.basename(x).split('_')[0][3:]))

	for image_path, mask_path in zip(images_path, masks_path):
		img = cv2.imread(image_path)
		mask = cv2.imread(mask_path, 0)
		ret, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
		
		label_img = label(mask)
		regions = regionprops(label_img)[0]
		minr,minc,maxr,maxc = regions.bbox

		quad_image = img[minr:maxr, minc:maxc, :]
		quad_h, quad_w = quad_image.shape[0], quad_image.shape[1]

		cv2.imwrite('tempo.png',quad_image)
		raw_detections = performDetect(imagePath='tempo.png', thresh= 0.15, configPath= configPath, weightPath = weightPath, metaPath= metaPath, showImage= showImage_flag)

		xmlfile = image_path[:-3] + 'xml'


		if showImage_flag:
			detections = raw_detections['detections']
			# detections2xml(detections['detections'], xmlfile, img, minr,minc,maxr,maxc)
			# plt.savefig('quad.png')
		else:
			detections = raw_detections

		detections2xml(detections, xmlfile, img, minr,minc,maxr,maxc)
		print(image_path)


