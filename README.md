Object localization using Convolutional neural networks to predict the bounding box co-ordinates
------------------------------------------------------------------------------------------------

This is a task wherein we have to predict the bounding box co-ordinates of the object present in the image. The dataset used for this project was the Flipkart Grid Challenge dataset (A competition held by FlipKart)

Dataset Description:
--------------------
   There were a total of 56,792 images in the dataset thus totalling to a size of about 14.2 GB of data. But in this task, along with the images we were also provided with the csv files containing the names of the images in the training and testing set. The training set had 24000 images whereas the testing set had 24045 images in it. Our task is to build a model capable of predicting the bounding box co-ordinates of the testing set images accuractely. 

Note: 
-----
   Since the dataset is big in size and due to some limitations , the dataset is not uploaded on git.   


Summary of the program files:
-----------------------------
1. The .m or the mat file was used to initially to preprocess the images. The image being a rgb color format was first converted to grayscale. Then a fast fourier transformation was applied to the image thus reducing it to its corrresponding frequency components. A high pass filter was used to filter out the low frequency components since the curves and edges correspond to high frequency. This new filtered mat array was now converted back to image and was rewritten in the images folder.

2. The yolo_new.py file consists of the main code for this localization problem. A 18 layer convolutional network similar to a yolo architecture was implemented. Initially the training data was loaded from the images folder with the help of the training csv file. The input data and the training labels were intially shuffled to better generalize over the data. The  The model was run for a total of 150 epochs and the model architecture as well as the weights were stored. A validation split of 0.1 was used on the training data.The final layer of the model employed the use of global average pooling to reduce the output to 4 units each corresponding to the bounding box co-ordinates. The optimizer used was rmsprop and the loss function used was mean squared error.

3. The yolo_pred.py file was  used to load the test data from the images folder. The saved architecture and weights were laoded and the model was now used to predict on the test data. The predicted bounding box labels were now saved to a csv file for submission.


Before submission, the predicted csv file was opened and the extra columns created by the dataframe was manually removed.(Index column)
