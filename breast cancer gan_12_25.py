#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os


# In[2]:


### importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
import cufflinks as cf
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
cf.go_offline()

### importing All models 
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from lightgbm import LGBMClassifier


# In[3]:


# dataset_path = 'C:/pythonn/breast_cancer_data/'

#C:\Users\sohel\Dropbox\Ariana\Interview_Data\Python_Model\Nirmal\Breast_cancer


# In[4]:


dataset_path = 'C:/Users/sohel/Dropbox/Ariana/Interview_Data/Python_Model/Nirmal/Breast_cancer/breast_cancer_data/'


# In[5]:


num_benign_images = len(os.listdir(os.path.join(dataset_path, 'BENIGN')))
num_malignant_images = len(os.listdir(os.path.join(dataset_path, 'MALIGNANT')))


# In[6]:


print("Number of benign images:", num_benign_images)
print("Number of malignant images:", num_malignant_images)


# In[7]:


import seaborn as sns
import matplotlib.pyplot as plt


labels = ['Benign', 'Malignant']
counts = [num_benign_images, num_malignant_images]

plt.figure(figsize=(10, 8))
sns.countplot(x=labels)
plt.xlabel('Image Type')
plt.ylabel('Count')
plt.title('Number of Benign and Malignant Images')

plt.show()


# In[9]:


benign_path = os.path.join(dataset_path, 'BENIGN')
malignant_path = os.path.join(dataset_path, 'MALIGNANT')


# In[9]:


benign_path


# In[10]:


benign_files = os.listdir(benign_path)


# In[11]:


import cv2


# In[12]:


fig, axes = plt.subplots(4, 4, figsize=(12, 12))
fig.suptitle('Benign Images', fontsize=16)
axes = axes.ravel()

for i, image_file in enumerate(benign_files[:16]):
    image_path = os.path.join(benign_path, image_file)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    axes[i].imshow(image)
    axes[i].axis('off')

plt.tight_layout()
plt.show()


# ###### for i, image_file in enumerate(benign_files[:9]):
#     image_path = os.path.join(benign_path, image_file)
#     image = cv2.imread(image_path)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     
#     # Find the type and size of the image
#     image_type = type(image)
#     image_size = image.shape
#     
#     # Display the image and print its type and size
#     axes[i // 3, i % 3].imshow(image)
#     axes[i // 3, i % 3].axis('off')
#     
#     print("Image", i+1)
#     print("Type:", image_type)
#     print("Size:", image_size)
#     print()
# 
# plt.tight_layout()
# plt.show()
# 

# In[15]:


malignant_files = os.listdir(malignant_path)


# In[17]:


fig, axes = plt.subplots(4, 4, figsize=(12, 12))
fig.suptitle('Malignant Images', fontsize=16)
axes = axes.ravel()

for i, image_file in enumerate(malignant_files[:16]):
    image_path = os.path.join(malignant_path, image_file)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    axes[i].imshow(image)
    axes[i].axis('off')

plt.tight_layout()
plt.show()


# In[18]:


image_shape = (28, 28, 3)
latent_dim = 100


# In[20]:


import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Reshape, Flatten, Conv2D, Conv2DTranspose
from keras.optimizers import Adam


# In[42]:


import numpy as np

# Create a random image with the specified shape
image_shape = (28, 28, 4)
image = np.random.randint(0, 256, size=image_shape, dtype=np.uint8)

# Print the shape and type of the image
print("Image shape:", image.shape)
print("Image type:", image.dtype)

# Display the image
import matplotlib.pyplot as plt
plt.imshow(image)
plt.axis('off')
plt.show()


# In[19]:


import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Reshape, Flatten, Conv2D, Conv2DTranspose
from keras.optimizers import Adam


# In[44]:


def build_generator():
    model = Sequential()
    model.add(Dense(7 * 7 * 256, input_dim=latent_dim))
    model.add(Reshape((7, 7, 256)))
    model.add(Conv2DTranspose(128, kernel_size=3, strides=2, padding='same'))
    model.add(Conv2DTranspose(64, kernel_size=3, strides=1, padding='same'))
    model.add(Conv2DTranspose(1, kernel_size=3, strides=2, padding='same', activation='tanh'))
    return model


# In[60]:


def build_discriminator():
    model = Sequential()
    model.add(Conv2D(64, kernel_size=3, strides=2, padding='same', input_shape=image_shape))
    model.add(Conv2D(128, kernel_size=3, strides=2, padding='same'))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    return model


# In[61]:


def build_gan(generator, discriminator):
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    return model


# In[62]:


def load_dataset():
    images = []
    classes = ['BENIGN', 'MALIGNANT']
    
    for class_name in classes:
        class_path = os.path.join(dataset_path, class_name)
        image_files = os.listdir(class_path)
        
        for image_file in image_files:
            image_path = os.path.join(class_path, image_file)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            
            if image is None:
                print(f"Failed to load image: {image_path}")
                continue
            
            image = cv2.resize(image, image_shape[:2])
            image = image.reshape(image_shape)
            image = image.astype(np.float32) / 255.0
            images.append(image)
    
    images = np.array(images)
    return images


# In[63]:


def load_dataset():
    images = []
    classes = ['BENIGN', 'MALIGNANT']
    
    for class_name in classes:
        class_path = os.path.join(dataset_path, class_name)
        image_files = os.listdir(class_path)
        
        for image_file in image_files:
            image_path = os.path.join(class_path, image_file)
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)  # Read the image with alpha channel
            
            if image is None:
                print(f"Failed to load image: {image_path}")
                continue
            
            image = cv2.resize(image, image_shape[:2])
            
            # If the image doesn't have an alpha channel, add one
            if image.shape[-1] == 3:
                image = np.concatenate([image, np.ones_like(image[:, :, :1]) * 255], axis=-1)
            
            image = image.astype(np.float32) / 255.0
            images.append(image)
    
    images = np.array(images)
    return images


# In[ ]:





# In[ ]:




def preprocess_images(images):
    processed_images = []
    for image in images:
        # Resize image to (28, 28)
        resized_image = cv2.resize(image, (28, 28))
        # Normalize image values to [-1, 1]
        normalized_image = resized_image / 255.0 * 2 - 1
        processed_images.append(normalized_image)
    processed_images = np.array(processed_images)
    return processed_images
# In[64]:


def preprocess_images(images):
    processed_images = []
    for image in images:
        # Resize image to (28, 28)
        resized_image = cv2.resize(image, (28, 28))
        
        # Ensure grayscale images have a third channel
        if len(resized_image.shape) == 2:
            resized_image = np.expand_dims(resized_image, axis=-1)
        
        # Normalize image values to [-1, 1]
        normalized_image = resized_image / 255.0 * 2 - 1
        processed_images.append(normalized_image)
    
    processed_images = np.array(processed_images)
    return processed_images


# In[65]:


def generate_fake_samples(generator, n_samples):
    noise = np.random.normal(0, 1, (n_samples, latent_dim))
    fake_images = generator.predict(noise)
    # Resize fake images to (28, 28)
    resized_fake_images = []
    for fake_image in fake_images:
        resized_fake_image = cv2.resize(fake_image, (28, 28))
        resized_fake_images.append(resized_fake_image)
    resized_fake_images = np.array(resized_fake_images)
    return resized_fake_images


# In[66]:


# Train the GAN
def train_gan(X_train, epochs, batch_size):
    # Build the generator and discriminator models
    generator = build_generator()
    discriminator = build_discriminator()
    
    # Compile the discriminator model
    discriminator.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    
    # Build the GAN model
    gan = build_gan(generator, discriminator)
    
    # Compile the GAN model
    gan.compile(loss='binary_crossentropy', optimizer=Adam())
    
    # Adversarial ground truths
    real = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))
    
    # Training loop
    for epoch in range(epochs):
        # Train the discriminator
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        real_images = X_train[idx]
        fake_images = generate_fake_samples(generator, batch_size)
        
        discriminator_loss_real = discriminator.train_on_batch(real_images, real)
        discriminator_loss_fake = discriminator.train_on_batch(fake_images, fake)
        discriminator_loss = 0.5 * np.add(discriminator_loss_real, discriminator_loss_fake)
        
        # Train the generator
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        generator_loss = gan.train_on_batch(noise, real)
        
        # Print the progress
        print(f"Epoch {epoch+1}/{epochs} - Discriminator Loss: {discriminator_loss[0]}, Generator Loss: {generator_loss}")
    
    # Save the generator model
    generator.save('generator_model.h5')


# In[67]:


images = load_dataset()

preprocessed_images = preprocess_images(images)

epochs = 200
batch_size = 32
train_gan(preprocessed_images, epochs, batch_size)


# In[68]:


from tensorflow import keras
# Generate fake images using the trained generator
def generate_images(generator, num_images):
    # Generate random noise as input for the generator
    noise = np.random.randn(num_images, 100)
    
    # Generate fake images
    generated_images = generator.predict(noise)
    
    return generated_images

# Load the trained generator model
generator = keras.models.load_model('generator_model.h5')

# Generate and display fake images
num_images = 9
fake_images = generate_images(generator, num_images)

# Rescale pixel values from [-1, 1] to [0, 1]
fake_images = (fake_images + 1) / 2.0

# Create a grid of subplots to display the fake images
fig, axes = plt.subplots(3, 3, figsize=(10, 10))
fig.suptitle('Generated Fake Images', fontsize=16)

# Plot each fake image in a subplot
for i, ax in enumerate(axes.flat):
    ax.imshow(fake_images[i], cmap='gray')
    ax.axis('off')

# Show the plot
plt.tight_layout()
plt.show()


# In[ ]:





# In[ ]:




