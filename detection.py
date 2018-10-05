from imageai.Detection import ObjectDetection
from PIL import Image
import os
import numpy as np
totalImages = 10

def resizeImg(width,height,imageFile,c):
	im1 = Image.open(imageFile)
	im5 = im1.resize((width, height), Image.ANTIALIAS)    # best down-sizing filter
	name = "resized"+str(c)+".jpg"
	im5.save(name)

execution_path = os.getcwd()
detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()

# for x in range(1,totalImages):
# 	imageFile = "dm"
# 	imageFile = imageFile + str(x) + ".jpg"
# 	resizeImg(600,600,imageFile,x)

#pat = np.array([[]])         ## 0,30,60,90,....540 can be starting coordinate of window, so (600/30) - 1
pat = []
for x in range(1,totalImages):
	imageFile = "resized"	
	imageFile = imageFile + str(x) + ".jpg"
	detections, feature_vect = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , imageFile), output_image_path=os.path.join(execution_path , "odm6.jpg"))
	#print(feature_vect)
	feature_vect = list(feature_vect)
	feature_vect.append(0)
	pat.append(feature_vect)
	#feature_vect = np.append(feature_vect, [0])
	#pat = np.appenbd(pat, [feature_vect], axis = 0)
	
for x in range(1,totalImages):
	imageFile = "resized"	
	imageFile = imageFile + str(x+9) + ".jpg"
	detections, feature_vect = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , imageFile), output_image_path=os.path.join(execution_path , "odm6.jpg"))
	#print(feature_vect)
	
	feature_vect = list(feature_vect)
	feature_vect.append(1)
	pat.append(feature_vect)

	# feature_vect = np.append(feature_vect, [1])
	# pat = np.append(pat, [feature_vect], axis = 0)

pat = np.array(pat)
np.save("training",pat)


# print(train)

# print(pat)

# for eachObject in detections:
# 	print(eachObject["name"] , " : " , eachObject["percentage_probability"] , " : " , eachObject["box_points"] )

	# 0 - dont move. 1 - move
	# sed -r 's/[[:space:]]+/,/g' testt.txt > testtc.txt
