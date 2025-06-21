# CIFAR-100 Image Classifier with Streamlit

This project allows you to classify images from the CIFAR-100 dataset with a trained TensorFlow CNN model via a Streamlit interface.

## Features

- ✅ CIFAR-100 trained model (100 distinct classes)
- ✅ Upload an image or capture one using webcam
- ✅ Real-time predictions directly from the browser
- ✅ Top-5 class predictions with confidence scores

---

## Project Structure

```bash
cifar100-classifier/
│
├── streamlit_app.py          # Streamlit web app
├── cifar10_cnn.py            # Model training script
├── cifar100_model_improved.keras  # Trained TensorFlow model (generated after training)
├── requirements.txt          # Python dependencies
└── README.md    
```
## Project documentation

### Setup Instructions
1️⃣ Install dependencies
Make sure you have Python 3.10+ installed.
```
pip install -r requirements.txt
```
2️⃣ Train the model
This step will download CIFAR-100, train a convolutional neural network (CNN), and save the model to disk.

```
python cifar10_cnn.py
```
After training, a file called cifar100_model_improved.keras will be generated.

3️⃣ Run the Streamlit app

```streamlit run streamlit_app_cifar100.py```

### How It Works
The app allows you to:

- Upload a local image file (JPG, JPEG, PNG), or

- Capture an image using your webcam (if supported in your browser)

The image is resized automatically to match CIFAR-100 input dimensions (32x32 pixels, RGB), normalized, and fed into the trained CNN model for prediction.

The model returns the top-5 most probable class names along with their confidence scores.

### Model Architecture
The neural network is a multi-layer CNN featuring:

- Convolutional layers with Batch Normalization

- MaxPooling layers

- Dropout for regularization

- Fully connected dense layers

- Final softmax output layer for 100-class classification

### CIFAR-100 Labels
The model classifies images into the 100 CIFAR-100 fine-grained labels:
```
apple, aquarium_fish, baby, bear, beaver, bed, bee, beetle, bicycle, bottle, bowl, boy, bridge, bus, butterfly, camel, can, castle, caterpillar, cattle, chair, chimpanzee, clock, cloud, cockroach, couch, crab, crocodile, cup, dinosaur, dolphin, elephant, flatfish, forest, fox, girl, hamster, house, kangaroo, keyboard, lamp, lawn_mower, leopard, lion, lizard, lobster, man, maple_tree, motorcycle, mountain, mouse, mushroom, oak_tree, orange, orchid, otter, palm_tree, pear, pickup_truck, pine_tree, plain, plate, poppy, porcupine, possum, rabbit, raccoon, ray, road, rocket, rose, sea, seal, shark, shrew, skunk, skyscraper, snail, snake, spider, squirrel, streetcar, sunflower, sweet_pepper, table, tank, telephone, television, tiger, tractor, train, trout, tulip, turtle, wardrobe, whale, willow_tree, wolf, woman, worm
```

### Requirements
- TensorFlow

- Streamlit

- Pillow

- OpenCV (for webcam input)