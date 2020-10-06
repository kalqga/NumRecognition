import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import Dropout,Flatten
from keras.layers.convolutional import Conv2D,MaxPooling2D
import pickle

# Variables
path = 'myData'
pathLabels = 'labels.csv'
test_ratio = 0.2
validation_ratio = 0.2
image_dimensions = (32, 32, 3)

batch_size_value = 50
epochs_value = 20
steps_per_epoch = 2000

####################
count = 0
images = []
class_number = []
myList = os.listdir(path)
print("Total number of classes detected: ", len(myList))
num_of_classes = len(myList)
print("Importing classes...")

for x in range(0, num_of_classes):
    my_pic_list = os.listdir(path + "/" + str(count))
    for y in my_pic_list:
        current_image = cv2.imread(path + "/" + str(count) + "/" + y)
        current_image = cv2.resize(current_image, (image_dimensions[0], image_dimensions[1]))
        images.append(current_image)
        class_number.append(count)
    print(count, end=" ")
    count += 1
print(" ")
print("Total images in images list = ", len(images))
print("Total IDS in class_number list = ", len(class_number))

images = np.array(images)
class_number = np.array(class_number)

####### Splitting the data
X_train, X_test, y_train, y_test = train_test_split(images, class_number, test_size=test_ratio)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_ratio)
print(X_train.shape)
print(X_test.shape)
print(X_validation.shape)

number_of_samples = []
for x in range(0, num_of_classes):
    number_of_samples.append(len(np.where(y_train == x)[0]))
print(number_of_samples)

plt.figure(figsize=(10, 5))
plt.bar(range(0, num_of_classes), number_of_samples)
plt.title("Number of images for each class")
plt.xlabel("Class ID")
plt.ylabel("Number of images")
plt.show()


def preProcessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255
    return img


# img = preProcessing(X_train[30])
# img = cv2.resize(img, (300, 300))
# cv2.imshow("PreProcessed", img)
# cv2.waitKey(0)

X_train = np.array(list(map(preProcessing, X_train)))
X_test = np.array(list(map(preProcessing, X_test)))
X_validation = np.array(list(map(preProcessing, X_validation)))

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
X_validation = X_validation.reshape(X_validation.shape[0], X_validation.shape[1], X_validation.shape[2], 1)

data_generator = ImageDataGenerator(width_shift_range=0.1,
                                    height_shift_range=0.1,
                                    zoom_range=0.2,
                                    shear_range=0.1,
                                    rotation_range=10)

data_generator.fit(X_train)

y_train = to_categorical(y_train, num_of_classes)
y_test = to_categorical(y_test, num_of_classes)
y_validation = to_categorical(y_validation, num_of_classes)


def myModel():
    number_of_filters = 60
    size_of_filter_1 = (5, 5)
    size_of_filter_2 = (3, 3)
    size_of_pool = (2, 2)
    number_of_nodes = 500

    model = Sequential()
    model.add((Conv2D(number_of_filters, size_of_filter_1, input_shape=(image_dimensions[0],
                                                                        image_dimensions[1],
                                                                        1),
                      activation='relu')))

    model.add((Conv2D(number_of_filters, size_of_filter_1, activation='relu')))
    model.add(MaxPooling2D(pool_size=size_of_pool))
    model.add((Conv2D(number_of_filters // 2, size_of_filter_2, activation='relu')))
    model.add((Conv2D(number_of_filters // 2, size_of_filter_2, activation='relu')))
    model.add(MaxPooling2D(pool_size=size_of_pool))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(number_of_nodes, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(num_of_classes, activation='softmax'))
    model.compile(Adam(lr=0.001), loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


model = myModel()
print(model.summary())


history = model.fit_generator(data_generator.flow(X_train, y_train,
                                        batch_size=batch_size_value),
                                        steps_per_epoch=steps_per_epoch,
                                        epochs=epochs_value,
                                        validation_data=(X_validation, y_validation),
                                        shuffle=1)

plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('Loss')
plt.xlabel('epoch')
plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'validation'])
plt.title('Accuracy')
plt.xlabel('epoch')
plt.show()
score = model.evaluate(X_test, y_test, verbose=0)
print('Test Score = ', score[0])
print('Test Accuracy = ', score[1])


pickle_out = open("A/model_trained.p", "wb")
pickle.dump(model, pickle_out)
pickle_out.close()
