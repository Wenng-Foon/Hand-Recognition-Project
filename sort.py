import os
import glob
import pdb
import cv2
import random

root = 'NUS/Hand Postures/'
label_dict = {'a':0, 'b':1, 'c':2, 'd':3, 'e':4, 'f':5, 'g':6, 'h':7, 'i':8, 'j':9}
if not os.path.exists(root+'train'):
	os.makedirs(root+'train')
if not os.path.exists(root+'val'):
	os.makedirs(root+'val')
train_img, val_img = [], []
img_list = glob.glob(root+'*.jpg')
length = len(img_list)

# for dirs in img_list:
	# img_list = glob.glob(dirs+'/*')
random.shuffle(img_list)
train_img.extend(img_list[:int(length*0.8)+1])
val_img.extend(img_list[int(length*0.8)+1:])
# pdb.set_trace()

for i,img in enumerate(train_img):
	# pdb.set_trace()
	label = label_dict[img.split('/')[-1][0]]
	img = cv2.imread(img)
	try:
		print(root+'train/'+str(label)+'_'+str(i).zfill(5)+'.jpg')
		cv2.imwrite(root+'train/'+str(label)+'_'+str(i).zfill(5)+'.jpg',img)
	except:
		print(root+'train/'+str(label)+'_'+str(i).zfill(5)+'.jpg')
for i,img in enumerate(val_img):
	label = label_dict[img.split('/')[-1][0]]
	img = cv2.imread(img)
	try:
		cv2.imwrite(root+'val/'+str(label)+'_'+str(i).zfill(5)+'.jpg',img)
	except:
		print(root+'val/'+str(label)+'_'+str(i).zfill(5)+'.jpg')
