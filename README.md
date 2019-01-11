# Project01
This project is for solving handwritting recognition problem. 
# DataSet
Handwritting Data provided by professor and Emnist.
# Requirement 
We use anaconda and jupter to design our project01 code.

The version of package used in this project is in the following.

(1)Numpy 1.15.4

(2)scikit-image 0.19.2

(3)OpenCV 3.4.2

(4)scikit-learn 0.14.0

# How to use it
In test.py, the default trainning dataset is called "Project01_Train_Dataset.npy" and it will be used as training dataset in the   model_generate function. If you want to change dataset for trainning, you need to chnage the path in it. 

In this model generate function, there are three types of model you can generate. 

If you put model_select=0 in model_generate(model_select=0), the model will be KNN method with Euclidean distance with K=2.

If you put model_select=1 in model_generate(model_select=1), the model will be KNN method with Manhattan distance with K=2.

If you put model_select=2 in model_generate(model_select=2), the model will be SVM method. 

Default generated model is KNN method with Euclidean distance. Recommended model is KNN method, the accuracy is about 99%. SVM has 90% accuracy. 

In cmd, you can type "python test.py Tested_data.npy out" for generating out.npy. out.npy is the outcome of Tested_data.npy.   








