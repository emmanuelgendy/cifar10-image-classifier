import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load CIFAR-10 again (to get the test set)
(_, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_test = x_test / 255.0  # Normalize

class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

# Load the saved model
model = tf.keras.models.load_model("cifar10_model.h5")

# Add softmax to output to get probabilities
probability_model = tf.keras.Sequential([
    model,
    tf.keras.layers.Softmax()
])


predictions = probability_model.predict(x_test)

def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i][0], img[i]
    plt.grid(False)
    plt.xticks([]); plt.yticks([])

    plt.imshow(img)

    predicted_label = np.argmax(predictions_array[i])
    color = 'blue' if predicted_label == true_label else 'red'

    plt.xlabel(f"{class_names[predicted_label]} ({100*np.max(predictions_array[i]):.2f}%)\n[True: {class_names[true_label]}]", color=color)

def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i][0]
    plt.grid(False)
    plt.xticks(range(10)); plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array[i], color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array[i])

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


num_rows = 5
num_cols = 3
num_images = num_rows * num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))

for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions, y_test, x_test)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions, y_test)

plt.tight_layout()
plt.show()

i = 7  # You can change this to any number between 0 and len(x_test)
plt.figure()
plot_image(i, predictions, y_test, x_test)
plt.show()
