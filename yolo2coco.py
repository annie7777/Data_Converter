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
	    content = d.split(" ")
	    filename = os.path.basename(d)[:-3]+'txt'
	    imgpath = os.path.join(root_dir, 'JPEGImages', os.path.basename(d))
	    # filename = content[0].split("/")[1]     #这里可能修改，txt文件每一行第一个属性是图片路径，通过split()函数把图像名分离出来就行
	    # img = cv2.imread(imgpath)
	    img = cv2.imread(imgpath)
	    try:
	        height,width = img.shape[0],img.shape[1]
	        image_id = int(imgid)+1
	    except:
	        times +=1
	        print('file is error')

	# type 已经填充

	#定义image 填充到images里面
	    image = {
	        'file_name' : os.path.basename(d),  #文件名
	        'height'    : height,    #图片的高
	        'width'     : width,     #图片的宽
	        'id'        : image_id   #图片的id，和图片名对应的
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
	            'area'          : area,  #
	            'iscrowd'       : 0,
	            'image_id'      : image_id,  #图片的id
	            'bbox'          :[xmin, ymin, o_width,o_height],
	            'category_id'   : int(category_id), #类别的id 通过这个id去查找category里面的name
	            'id'            : bnd_id,  #唯一id ,可以理解为一个框一个Id
	            'ignore'        : 0,
	            'segmentation'  : []
	        }

	        json_dict['annotations'].append(annotation)

	        bnd_id += 1
	    #
	#定义categories

	#你得类的名字(cid,cate)对应
	# classes = ['0','1','2','3','4','5','6','7','8','9']

	with open(root_dir+'/classes.names','r') as f:
	    classes = f.read().splitlines()

	for i in range(len(classes)):

	    cate = classes[i]
	    cid = i
	    category = {
	        'supercategory' : 'phenology',
	        'id'            : cid,  #类别的id ,一个索引，主键作用，和别的字段之间的桥梁
	        'name'          : cate  #类别的名字比如房子，船，汽车
	    }

	    json_dict['categories'].append(category)



	json_fp = open(os.path.join(root_dir,"{}.json".format(txtfile[:-4])),'w')
	json_str = json.dumps(json_dict)
	json_fp.write(json_str)
	json_fp.close()