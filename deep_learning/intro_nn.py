import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras


#First get the dataset from the keras database and split it on a train and test set
mnist = keras.datasets.mnist
(X_train_full,y_train_full),(X_test,y_test) = mnist.load_data()

#Split the training set in a new reduced train set and a validation set 
#Scale the validation and training sets dividing by 255 which is the maximum pixel intensity

X_valid, X_train = X_train_full[:50000]/255.0 , X_train_full[50000:]/255.0
y_valid, y_train = y_train_full[:50000] , y_train_full[50000:]
X_test = X_test/255.0


#Plot learning curves
def plt_learning_curves(model_history):
    pd.DataFrame(model_history.history).plot(figsize = (8,5))
    plt.grid(True)
    plt.gca().set_ylim(0,1)
    plt.show()

#Simple sequential API approach:
def simple_sequential():

    #Constructing sequential model
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape = [28,28]))
    model.add(keras.layers.Dense(300, activation = 'relu'))
    model.add(keras.layers.Dense(100, activation = 'relu'))
    model.add(keras.layers.Dense(10, activation = 'softmax'))
    model.summary()

    #Compiling the model
    model.compile(loss = 'sparse_categorical_crossentropy',
            optimizer = 'sgd',
            metrics = ['accuracy'])

    #Training the model   
    checkpoint_cb = keras.callbacks.ModelCheckpoint('keras_simp_seq_model.h5', save_best_only = 'True')
    history = model.fit(X_train, y_train,
            epochs = 30,
            validation_data = (X_valid, y_valid),
            callbacks = [checkpoint_cb])
    
    #Plotting learning curves
    plt_learning_curves(history)

    return model

if __name__ == '__main__':
    
    print('Simple Sequential API approach')
    simple_model = simple_sequential()

    print('Evaluating Simple Sequential API model on the test set')
    simple_model.evaluate(X_test,y_test)
 


