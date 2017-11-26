import cv2
import numpy as np
import os         
from random import shuffle 
from tqdm import tqdm      
import tensorflow as tf
import matplotlib.pyplot as plt
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

TRAIN_DIR = 'CRCHistoPhenotypes_2016_04_28/TrainNuclei'
TEST_DIR = 'CRCHistoPhenotypes_2016_04_28/TrainNuclei'
IMG_SIZE = 25
LR = 1e-3
MODEL_NAME = 'nuclei-convnet'

def create_label(image_name):
    """ Create an one-hot encoded vector from image name """
    word_label = image_name.split('.')[0]
    if word_label == 'notnucleus':
        return np.array([1,0])
    elif word_label == 'nucleus':
        return np.array([0,1])

def create_train_data():
    training_data = []
    for img in tqdm([f for f in os.listdir(TRAIN_DIR) if not f.startswith('.')]):
        path = os.path.join(TRAIN_DIR, img)
        img_arr = img.split('.')
        if img_arr[-2] != '1':
            img_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
            training_data.append([np.array(img_data), create_label(img)])
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data

def create_test_data():
    testing_data = []
    for img in tqdm([f for f in os.listdir(TRAIN_DIR) if not f.startswith('.')]):
        path = os.path.join(TEST_DIR,img)
        img_arr = img.split('.')
        if img_arr[-2] == '1':
            img_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
            testing_data.append([np.array(img_data), create_label(img)])
    shuffle(testing_data)
    np.save('test_data.npy', testing_data)
    return testing_data

# If dataset is not created:
# train_data = create_train_data()
# test_data = create_test_data() 
# If you have already created the dataset:
train_data = np.load('train_data.npy')
test_data = np.load('test_data.npy')
train = train_data
test = test_data
# X_train = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
# y_train = [i[1] for i in train]
# X_test = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
# y_test = [i[1] for i in test]

tf.reset_default_graph()
convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')
convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 128, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)
convnet = fully_connected(convnet, 2, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')
model = tflearn.DNN(convnet, tensorboard_dir='log', tensorboard_verbose=0)
# model.fit({'input': X_train}, {'targets': y_train}, n_epoch=10, 
#           validation_set=({'input': X_test}, {'targets': y_test}), 
#           snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

# model.save(MODEL_NAME)

model.load(MODEL_NAME)


fig=plt.figure()

for num,data in enumerate(test_data[24:36]):
    # notnucleus: [1,0]
    # nucleus: [0,1]
    img_num = data[1]
    img_data = data[0]
    
    y = fig.add_subplot(3,4,num+1)
    orig = img_data
    data = img_data.reshape(IMG_SIZE,IMG_SIZE,1)
    #model_out = model.predict([data])[0]
    model_out = model.predict([data])[0]
    print(img_num)
    print("Not Nucleus:", model_out[0], "Nucleus:", model_out[1])
    
    if np.argmax(model_out) == 1: str_label='Nucleus'
    else: str_label='Not Nucleus'
        
    y.imshow(orig, cmap="gray")
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
plt.show()
