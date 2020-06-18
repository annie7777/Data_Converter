# Data_Converter

## 1. [yolo2coco](yolo2coco.py)

folder structure
 
    .
    ├── yolo                    # root dir
    │   ├── JPEGImages          # image dir
    │       ├── img1.png
    │       ├── img2.png
    │       └── ...
    │   ├── labels              # label dir
    │       ├── img1.txt        # label txt file: label(int) xc yc w h
    │       ├── img2.txt
    │       └── ...  
    │   ├── train.txt           # train list
    │   ├── test.txt            # test list
    │   └── classes.names       # category
    └── ...
    
 
## 2. [yolo2labelimg](yolo2labelimg.py)

update yolo configuration block and quad configuration block