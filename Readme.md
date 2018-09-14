Summary
===== 

This is Mati's exam implementation using Keras. __The result is 69% of the test captchas were correct.__ The main trick was to augment the training data, as well as using augmented test images statistics to further improve the evaluation. As well as obvious resolution decrease for the network to be able to train. 

__Note:__ Reached 76% using 250 epochs instead of 100, and 78% With adding extra 40K augmentations and another 200 epochs. 

Steps to reproduce: 
====== 

0. Download the repository to your local drive. 
1. Copy labeled capcha to repository into directory 'data/labeled_capcha' on you local drive. 
2. Run data_preproccess.py - fix the issues. Check 'data' directory, it should have 4 new directories filled with images. 
3. Run train.py to train. 
4. Run test.py to get report of predicted test images. The final line in the report shows the percentile of succesful captcha. 

Working progress
=======

0. For starters I wanted to see what happens when I train a network on the data as is. So I have an array of images as input and the output 
is just the first digit/letter. 

- As we have only 800 images to train with, and the images are large (330x150). In order to find a compromise I resized the images to 66x30. I got 
something i.e. validation accuracy of 27%.

1. Now as the obvious problem is that there is not enough data as is. So the most obvious solution is augmentation. As we have only shifts, 
no rotation and scale, the augmentation will be useful to also have some sort of quantity of the data, that will allow more stable networks. 

- in order to use augmentation I splitted the data to test and train. I augmented the train only for training. 

- Another idea came to my mind, is that even if I have a network which works in 50% of the time on the augmented data. I can take test image, augment it 100 times and hope the majority of times the answer will be correct (the mistakes are noisy and the correct answer is consistent). So if the simple approach will give somewhat good results I will use this approach. 

- I've trained a network which has a problem to guess the first digit (around 50%) the other digits are at 90%, on the same type of data (i.e. augmented). Now the plan to generate augmented data for the test. 

- I've generated data for the test. Got 69% success for the whole captcha, which I think is not that bad considering we have small amount of examples, and the time allocated to solve the problem. 


Notes
==== 

- Although this approach is not using the whole robustness of the input resolution, the augmentation is actually solving this issue, because many augmented images with smaller resolution should contain the information about the original image (i.e. it can be reconstructed using super-resolution), so the high resolution information is not lost but transformed. In my case I lost some of the resolution because of disk-space issues (but I didn't resize to 66x30 only to 165 to 75 while augmenting).

- Improving it further might helpped by some more augmented images. This is not the best approach but it could help further. 

- Other approaches could include artificially generating more similiar data (for example by using GAN or writing code to generate captchas). 
