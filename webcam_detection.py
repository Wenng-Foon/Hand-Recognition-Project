
import os
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
from modelresnet10 import resnet10, CBAM_resnet10


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CBAM_resnet10(1, channels=3, reduction_ratio=16, pool_type="avg", use_Spatial=True)
checkpoint_path = './res10_cbam_checkpoint.pth'
assert os.path.exists(checkpoint_path), ('checkpoint do not exits for %s' % checkpoint_path)
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
    #transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                     std=[0.229, 0.224, 0.225])
])

font                   = cv2.FONT_HERSHEY_SIMPLEX
top                    = (20,20)
bottomLeft             = (400,400)
fontScale              = 1
fontColor              = (255,255,255)
lineType               = 2
prev_frame_time = time.time()
with torch.no_grad():

    cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    
    while cap.isOpened():

        ret, frame = cap.read()
        
        #img = frame
        img = cv2.resize(frame, (180,160))
        img = transform(img)
        img = img.to(device)
        img = img.unsqueeze(0)
        
        # Make detections 
        results = model(img)
        score = results.item()
        
        # FPS calc
        new_frame_time = time.time()
        fps = round(1/(new_frame_time-prev_frame_time))
        prev_frame_time = new_frame_time
        
        img = img.cpu()
        img = img[0].numpy().transpose(1, 2, 0)
        
        cv2.putText(frame, "Detection score: {}".format(score), top, font, fontScale, fontColor, lineType)
        cv2.putText(frame, "FPS: {}".format(fps), bottomLeft, font, fontScale, fontColor, lineType)

        cv2.imshow('Hand Detection', frame)
    
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
