
import argparse
import os
import numpy as np

import tensorflow.keras
from tensorflow.keras.layers import Flatten, Input, concatenate, Dense, Activation, Dropout, BatchNormalization,  MaxPooling2D, AveragePooling2D, Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG19


from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


from azureml.core import Run

from numpy.random import seed
seed(1)
from tensorflow.random import set_seed
set_seed(2)

# let user feed in 4 parameters:
# the datasets of the generated and test data (mount or download);
# the amount of epochs to train;
# the batch size to train;
# The model name
parser = argparse.ArgumentParser()
parser.add_argument('--data-folder', type=str, dest='data_folder', help='Test data folder mounting point')
parser.add_argument('--epochs', type=str, dest='epochs', help='Amount of epochs to train')
parser.add_argument('--batch_size', type=str, dest='batch_size', help='Batch size')
parser.add_argument('--model_name', type=str, dest='model_name', help='Model name')
args = parser.parse_args()


data_folder = args.data_folder
print('Data folder:', data_folder)

X_test_conv = np.load(os.path.join(data_folder, 'X_test_conv.npy'))
X_test_values = np.load(os.path.join(data_folder, 'X_test_values.npy'))
y_test = np.load(os.path.join(data_folder, 'y_test.npy'))

X_train_conv = np.load(os.path.join(data_folder, 'X_train_conv_generated.npy'))
X_train_values = np.load(os.path.join(data_folder, 'X_train_values_generated.npy'))
y_train = np.load(os.path.join(data_folder, 'y_train_generated.npy'))

print(X_test_conv.shape, X_test_values.shape, y_test.shape, sep = '\n')
print(X_train_conv.shape, X_train_values.shape, y_train.shape, sep = '\n')

labels = np.load(os.path.join(data_folder, 'component_names.npy')).tolist()
print(f"Labels: {labels}")

# get hold of the current run
run = Run.get_context()

# Neural network
batch_size = int(args.batch_size)
epochs = int(args.epochs)
X_conv = Input(shape=(64, 64, 3))

vgg_model = VGG19(include_top=False, weights='imagenet')(X_conv)    # Add all the layers of the VGG19 model

x_1 = Flatten(name='flatten')(vgg_model)
x_1 = Dense(512, activation='relu', name='fully-connected-1')(x_1)
x_1 = Dense(512, activation='relu', name='fully-connected-2')(x_1)

X_extra = Input(shape=(1,))

combined = concatenate([x_1, X_extra]) ## Combined input
x_2 = Dense(16, activation='relu', name='combined-fully-connected-1')(combined)
## Output
x_2 = Dense(y_test.shape[1], activation='softmax', name='combined-fully-connected-2')(x_2)

final_model = Model(inputs=[X_conv, X_extra], outputs=x_2)

opt = tensorflow.keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0, clipvalue=0.6)

final_model.compile(optimizer=opt, loss='categorical_crossentropy', 
                   metrics=['accuracy'])

early_stopping_callback = tensorflow.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5)
reduce_lr = tensorflow.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1)


history = final_model.fit([X_train_conv, X_train_values],
                            y_train,
                            validation_data=([X_test_conv, X_test_values], y_test),
                            epochs = epochs,
                            batch_size = batch_size, # x * batch_size == amount of data in one epoch
                            verbose = 1,
                            shuffle=True,
                            workers=1,
                            callbacks=[reduce_lr]
                        )

print('Predict the test set')
predictions = final_model.predict([X_test_conv, X_test_values])
predictions = predictions.argmax(axis=1)

class_report = classification_report(y_test.argmax(axis=1), predictions, target_names=labels)
run.log("Classification report", class_report)

cf = confusion_matrix(y_test.argmax(axis=1), predictions)
# run.log_confusion_matrix("Confusion Matrix", cf)
run.log("Confusion matrix", cf)

acc = accuracy_score(y_test.argmax(axis=1), predictions) * 100


run.log('accuracy', np.float(acc))
run.log("Epochs:", epochs)
run.log("batch_size", batch_size)

run.log_list("Accuracy", history.history['accuracy'])
run.log_list("Validation accuracy", history.history['val_accuracy'])
run.log_list("Loss", history.history['loss'])
run.log_list("Validation loss", history.history['val_loss'])

os.makedirs('outputs', exist_ok=True)
# note file saved in the outputs folder is automatically uploaded into experiment record
final_model.save(f"outputs/{str(args.model_name)}")
