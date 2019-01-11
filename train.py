import numpy as np 
from skimage.feature import hog 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score 
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC 
import cv2 
import pickle
from sklearn.model_selection import train_test_split 
def generate_data(dataset=np.load('Project01_Train_Dataset.npy'),labels=np.load('Project01_Train_Dataset_Label.npy')):
    labels = np.concatenate((labels,labels), axis=None)
    labels = np.concatenate((labels,labels), axis=None)
    datalist = []
    for j in range (0,dataset.size):
        a = cv2.resize(dataset[j],(28,28))
        datalist.append(a)
    for j in range (0,dataset.size):
        b = cv2.resize(dataset[j],(28,28),interpolation=cv2.INTER_AREA)
        datalist.append(b)
    for j in range (0,dataset.size):
        c = cv2.resize(dataset[j],(28,28),interpolation=cv2.INTER_CUBIC)
        datalist.append(c)
    for j in range (0,dataset.size):
        d = cv2.resize(dataset[j],(28,28),interpolation=cv2.INTER_LANCZOS4)
        datalist.append(d)
    return np.array(datalist),labels

def processd_data_with_hog(data,labels,Train=True):
    if Train ==True:
        train_data, test_data, train_labels, test_labels = train_test_split(data,labels, test_size=0.3)
        test_hogimagelist = [] 
        test_hoglist = [] 
        hogimagelist = [] 
        hoglist = [] 
        for c in range (0,test_data.shape[0]): 
            fd1, hog_image1 = hog(test_data[c], orientations = 8,pixels_per_cell=(4, 4),cells_per_block=(1, 1), visualize=True, multichannel=False) 
            test_hoglist.append(fd1.T) 
            test_hogimagelist.append(hog_image1)
        test_hog = np.array(test_hoglist)
        for j in range (0,train_data.shape[0]): 
            fd, hog_image = hog(train_data[j], orientations= 8, pixels_per_cell=(4, 4),cells_per_block=(1, 1), visualize=True, multichannel=False) 
            hoglist.append(fd.T) 
            hogimagelist.append(hog_image) 
        train_hog = np.array(hoglist) 
    else:
        test_hog=np.load('test_hog.npy')
        train_hog=np.load('train_hog.npy')
        train_labels=np.load('train_labels.npy')
        test_labels=np.load('test_labels.npy')
    return test_hog,train_hog,train_labels,test_labels
def model_generate(data_input=np.load('Project01_Train_Dataset.npy'),Label_input=np.load('Project01_Train_Dataset_Label.npy'),model_select=0,Train=False):
    data, labels = generate_data(data_input,Label_input)
    test_hog,train_hog,train_labels,test_labels=processd_data_with_hog(data,labels,Train)
    metric = ['euclidean','manhattan']
    k=2
    if model_select==2 and Train==True:
        with open('clf_SVM_Method.pickle','wb') as f:
            clf = SVC(gamma = 0.01)
            clf.fit(train_hog, train_labels)
            pickle.dump(clf,f)
            
        return clf
    elif model_select==2 and Train==False:
        with open('clf_SVM_Method.pickle','rb') as f:
            clf=pickle.load(f)
            
        return clf
    
    elif model_select==0 and Train==True: 
        classifier = KNeighborsClassifier(n_neighbors=k,weights = 'distance',metric = metric[0])
        classifier.fit(train_hog, train_labels)
        with open('clf_KNN_Method_Euclidean.pickle','wb') as f:
            pickle.dump(classifier,f)
        
        return classifier
    elif model_select==0 and Train==False:
        with open('clf_KNN_Method_Euclidean.pickle','rb') as f:
            return pickle.load(f)
    
    elif model_select==1 and Train==True: 
        classifier = KNeighborsClassifier(n_neighbors=k,weights = 'distance',metric = metric[1])
        classifier.fit(train_hog, train_labels)
        with open('clf_KNN_Method_Manhattan.pickle','wb') as f:
            pickle.dump(classifier,f)
        return classifier
    elif model_select==1 and Train==False:
        with open('clf_KNN_Method_Manhattan.pickle','rb') as f:
            return pickle.load(f)
