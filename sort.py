import os
import glob
import pdb
import cv2
import random

root = 'NUS'
if not os.path.exists('NUS/train'):
	os.makedirs('NUS/train')
if not os.path.exists('NUS/val'):
	os.makedirs('NUS/val')
train_img, val_img = [], []
dir_list = glob.glob(root+'/*')
for dirs in dir_list:
	img_list = glob.glob(dirs+'/*')
	random.shuffle(img_list)
	length = len(img_list)
	train_img.extend(img_list[:int(length*0.8)+1])
	val_img.extend(img_list[int(length*0.8)+1:])

for i,img in enumerate(train_img):
	img = cv2.imread(img)
	# if not i < 1191:
	# 	continue 
	try:
		cv2.imwrite('NUS/train/'+str(i).zfill(4)+'.jpg',img)
	except:
		print('NUS/train/'+str(i).zfill(4)+'.jpg')
	# os.system('scp {} NUS/train/{}.jpg'.format(img,str(i).zfill(4)))
for i,img in enumerate(val_img):
	img = cv2.imread(img)
	try:
		cv2.imwrite('NUS/val/'+str(i).zfill(4)+'.jpg',img)
	except:
		print('NUS/val/'+str(i).zfill(4)+'.jpg')
