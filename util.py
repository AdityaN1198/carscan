from PIL import ImageDraw, Image
import cv2 as cv
import os
import numpy as np
from numpy.core.numeric import isclose
import json


def label_img(meta_data_json_path,image_path,opacity =0.7, display_cv=False):
	# Input the json file directory, image file directory and opacity  level ranging between 0 to 1. If debug is true then no image will be displayed.

	# intiate an empty dictionary and store all json data to key 'objects' and save image directory to key 'imagePath'
	data = {}

	with open(meta_data_json_path) as f:
		json_data =  json.load(f)

	data['objects'] = json_data
	data['imagePath'] = str(image_path)

	# load the image in OpenCV and get its height and weight
	img = cv.imread(data["imagePath"])
	height, width = img.shape[:2]

	#color for fill and line
	colors = [(215,0,0),(0,215,0),(0,0,215),(117,23,123),(224,84,0),(224,201,0),(191,50,73)]
	colors_for_lines = [(255,0,0),(0,235,0),(0,0,235),(157,23,123),(224,124,0),(224,241,0),(191,90,73)]
	
	# iterate through each key in the json data and get the points and labels
	
	line_img = img.copy()
	for i in range(len(data['objects'])):
		
		if data['objects'][i]['type'] =="polygonlabels":
			
			# List to save all the points
			points = []

			# copies of the image to apply masks on
			overlay = img.copy()

			# variables to record last points of polygon so they can be used to draw bounding boxs
			min_xcord = float('inf')
			max_xcord = float('-inf')
			min_ycord = float('inf')
			max_ycord = float('-inf')

			# Extract label of the part
			label = data['objects'][i]['value']["polygonlabels"][0]
			
			# iterate through points and save them to dictionary as [x,y] coordinates
			for point in data['objects'][i]['value']['points']:

				# denormalized points
				x_cord = (point[0]*width)/100
				y_cord = (point[1]*height)/100

				points.append([x_cord,y_cord])

				# updating value of last found point
				if x_cord < min_xcord:
					min_xcord = round(x_cord)
				elif x_cord > max_xcord:
					max_xcord = round(x_cord)

				if y_cord < min_ycord:
					min_ycord = round(y_cord)
				elif y_cord > max_ycord:
					max_ycord = round(y_cord)

			# convert points to numpy array to use in OpenCV masking
			points = np.array(points)

			# get text size to form blackbackground
			text_size, _ = cv.getTextSize(label,fontFace=cv.FONT_HERSHEY_PLAIN , fontScale=0.8,thickness=10)
			text_w, text_h = text_size
			
			# Draw polygon on the copied imaeg
			cv.fillPoly(overlay,np.int32([points]),color=colors[i%len(colors)])
			# Draw boundary of polygon
			cv.polylines(overlay,np.int32([points]),isClosed=False,color=colors_for_lines[i%len(colors_for_lines)],thickness=4)
			# Draw rectangle around polygon based on last points of polygon
			cv.rectangle(overlay,(min_xcord,min_ycord),(max_xcord,max_ycord),color=colors_for_lines[i%len(colors_for_lines)],thickness=2)

			# Draw black background for text
			cv.rectangle(overlay,(min_xcord,min_ycord),(min_xcord+text_w,min_ycord+text_h+10),(0,0,0),-1)

			# Put label to the corresponding bounding box
			cv.putText(img=overlay,text=label,org=(min_xcord,min_ycord+15),fontFace=cv.FONT_HERSHEY_PLAIN,fontScale=0.8,color=(255,255,255))
			
			# Set opacity  provided in the function. 0.7 by default
			alpha = opacity 

			# Blend the original image with masked image
			img = cv.addWeighted(overlay, alpha, img, 1-alpha,0)
			
			cv.polylines(line_img,np.int32([points]),isClosed=False,color=colors_for_lines[i%len(colors_for_lines)],thickness=3)
			

	

	if display_cv:
		cv.imshow('final',img)
		cv.imshow('lines',line_img)
		cv.waitKey(0)
		cv.destroyAllWindows()
	
	return img, line_img


