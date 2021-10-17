NUS dataset I & II 
https://www.ece.nus.edu.sg/stfpage/elepv/NUS-HandSet/
- 2000 Static hand images (low res of 160 x 120) of large background variety
- 2000 empty background images
- 750 hand images with people in background
- easier to do (only 100MB total)
- train a detector(using resnet10) with the 2000 empty background dataset
- test out resnet50 for recognition


IPN dataset 
Homepage: https://gibranbenitez.github.io/IPN_Hand/
Paper: https://gibranbenitez.github.io/2021_ICPR_IpnHand.pdf
- large gesture (series of motion images) dataset for interactive applications
- more difficult as large data and model


TASKS TO DO: (Single frame recognition)
1. Perform data augmentation via dataloader transforms (rotation, scale, horizontal flip, translation)
2. Make a simple detector with resnet10, fc layer maps to 2 classes: hand present or not present
3. Make a complex recognition model with resnet50 and other architectures to perform hand recognition 


Extension: (Multiframe recognition) 
1. Preprocess IPN dataset into categories of frames of the 11 classes (including no hand)
2. Convert all image into greyscale, stack 5 of the same images together to create a 5 channel input
3. Put these 5 channel input through a ResNext101 (bottleneck resnet) and train with label supervision
4. Perform predictiong and evaluate test accuracy 

Implementation details:
When the detector detects a hand for a certain number of times in a time period of video capture, it will turn on the recognition model.

*Feel free to start the coding, we can share later via google drive or if you want, drop me your email for your github account and i will give you access to upload code here.
