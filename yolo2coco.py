import json
import cv2
import os

# yolo root dir
root_dir = 'yolo'
txtfiles = ['train.txt', 'test.txt']

bnd_id_start = 1
times = 0
json_dict = {
	"images"     : [],
	"type"       : "instances",
	"annotations": [],
	"categories" : []
}

for txtfile in txtfiles:

	with open(os.path.join(root_dir,txtfile),'r') as f:
		data = f.read().splitlines()

	bnd_id = bnd_id_start

	for imgid, d in enumerate(data):
		filename = os.path.basename(d)[:-3]+'txt' #yolo label txt file 
		imgpath = os.path.join(root_dir, 'JPEGImages', os.path.basename(d)) # img file
		img = cv2.imread(imgpath)
		try:
			height,width = img.shape[0],img.shape[1]
			image_id = int(imgid)+1
		except:
			times +=1
			print('file is error')

		image = {
			'file_name' : os.path.basename(d), 
			'height'    : height,   
			'width'     : width,     
			'id'        : image_id  
		}
		json_dict['images'].append(image)


		with open(root_dir+'/labels/'+filename,'r') as f:
			labels = f.read().splitlines()


		for c in labels:
			label,xc,yc,w,h = c.strip().split(" ")
			xmin = float(xc)*width - float(w)*width/2
			ymin = float(yc)*height - float(h)*width/2
			xmax = float(xc)*width + float(w)*width/2
			ymax = float(yc)*height + float(h)*width/2
			xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
			o_width = abs(int(xmax) - int(xmin))
			o_height = abs(int(ymax) - int(ymin))

			area = o_width * o_height
			category_id = label.strip()

			# #定义annotationhb
			annotation = {
				'area'          : area,  
				'iscrowd'       : 0,
				'image_id'      : image_id, 
				'bbox'          :[xmin, ymin, o_width,o_height],
				'category_id'   : int(category_id), 
				'id'            : bnd_id, 
				'ignore'        : 0,
				'segmentation'  : []
			}

			json_dict['annotations'].append(annotation)

			bnd_id += 1

	# yolo class file
	with open(root_dir+'/classes.names','r') as f:
		classes = f.read().splitlines()

	for i in range(len(classes)):

		cate = classes[i]
		cid = i
		category = {
			'supercategory' : 'phenology',
			'id'            : cid, 
			'name'          : cate  
		}

		json_dict['categories'].append(category)



	json_fp = open(os.path.join(root_dir,"{}.json".format(txtfile[:-4])),'w')
	json_str = json.dumps(json_dict)
	json_fp.write(json_str)
	json_fp.close()