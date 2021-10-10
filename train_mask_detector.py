# Package import
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os



# Initializing the initial learning rate, number of epochs to train for, & batch size
# Learning rate is less so that loss is calculated properly; to get better accuracy soon.
INIT_LR = 1e-4 #0.0001
# Epochs: number times that the learning algorithm will work through the entire training dataset.
EPOCHS = 20
# The batch size is a number of samples processed before the model is updated.
BS = 32


#---Data pre-processing, converting images to arrays->Create DL model

#grabbing and categorising data
DIRECTORY = r"D:\Code\Face-Mask-Detection-master\dataset"
CATEGORIES = ["with_mask", "without_mask"]

# grabbing the list of images in our dataset directory, then initialize
# the list of data (i.e., images) and class images
print("[INFO] loading images...")

#Appending corresponding image and labels arrays in list
data = []
labels = []

#Loop through 2 categories i.e. with/without mask in dataset directory
for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category)

	#listdir method list down all the images
    for img in os.listdir(path):

		#join path with corresponding image
    	img_path = os.path.join(path, img)

		#From Keras, load image and target size: height and width for image uniformity
    	image = load_img(img_path, target_size=(224, 224))

		#From keras, image to Array
    	image = img_to_array(image)

		#Using Mobile net for this model requries preprocess_input
    	image = preprocess_input(image)

    	data.append(image)
    	labels.append(category)

# perform one-hot encoding on the labels

#converting label text to numerical values
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

#Converting list into array via NumPy
data = np.array(data, dtype="float32")
labels = np.array(labels)

#Splitting training and testing data, 20% for testing, 80% for training.
#Random state ensures that the splits that you generate are reproducible.
(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.20, stratify=labels, random_state=42)


#---Training
# construct the training image generator for data augmentation

# Augmentation IDG: Creates many images by flipping/rotating the image, for more dataset
aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

# load the MobileNetV2 network, ensuring the head FC layer sets are left off
# MobileNet: Faster and takes lesser parameteres
# ImageNet is pre-trained models which has weights, when initalized it will give better accuracy
# include_top = false to make weight compatible as any change can make it incompatible
baseModel = MobileNetV2(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))

# construct the head of the model that will be placed on top of the
# the base model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
#128 Dense layers; ReLU (Rectified Linear Unit) activation function is an activation function defined as the positive part of its argument: where x is the input to a neuron.
headModel = Dense(128, activation="relu")(headModel)
#Dropout to avoid overfitting of our model
headModel = Dropout(0.5)(headModel)
#Softmax is an activation function for binary classication of images
headModel = Dense(2, activation="softmax")(headModel)

# place the head FC model on top of the base model (this will become
# the actual model we will train)
model = Model(inputs=baseModel.input, outputs=headModel)

# loop over all layers in the base model and freeze them so they will
# *not* be updated during the first training process
for layer in baseModel.layers:
	layer.trainable = False

# compile our model
print("[INFO] compiling model...")
#training deep learning models, Adam is the best
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
#Binary cross entropy compares each of the predicted probabilities to actual class output which can be either 0 or 1
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# train the head of the network
#train more data as we have less data
print("[INFO] training head...")
H = model.fit(
	aug.flow(trainX, trainY, batch_size=BS),
	steps_per_epoch=len(trainX) // BS,
	validation_data=(testX, testY),
	validation_steps=len(testX) // BS,
	epochs=EPOCHS)

# make predictions on the testing set
print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)

# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
predIdxs = np.argmax(predIdxs, axis=1)

# show a nicely formatted classification report
print(classification_report(testY.argmax(axis=1), predIdxs,
	target_names=lb.classes_))

# serialize the model to disk
print("[INFO] saving mask detector model...")
model.save("mask_detector.model", save_format="h5")

# plot the training loss and accuracy
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")