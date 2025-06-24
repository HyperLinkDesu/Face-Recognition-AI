#"put #%% to run on jupyter"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
import matplotlib.pyplot as plt                            #to plot  
import numpy as np



# helped by https://www.youtube.com/watch?v=wQ8BIBpya2k
#           https://www.tensorflow.org/tutorials/quickstart/beginner



mnist = keras.datasets.mnist # 28x28 images of hand written digits 0 to 9

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

#print(x_train[0]) #print the tensor 0

#plt.imshow(x_train[0], cmap= plt.cm.binary) 
#plt.show()                                 use jupyter directory to graph the image

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])

predictions = model(x_train[:1]).numpy()

tf.nn.softmax(predictions).numpy()

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

loss_fn(y_train[:1], predictions).numpy()


model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])


model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test,  y_test, verbose=2)


probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
prediction = probability_model.predict(x_test[:5])
print("\nPredictions for first 5 test images:")
# print(prediction) # prints the probablity distribution

for i in range(len(prediction)):
    predicted_class = np.argmax(prediction[i])
    print(f"Image {i+1}: Predicted class = {predicted_class}")
    plt.imshow(x_test[i], cmap= plt.cm.binary) 
    plt.show()


#%%

#%%