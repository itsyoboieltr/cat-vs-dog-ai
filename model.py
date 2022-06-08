from tensorflow.keras import layers, optimizers, models, Model
import tensorflowjs as tfjs
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import numpy as np
from os import path, rename, mkdir, remove
import requests
from tqdm import tqdm
from kaggle.api.kaggle_api_extended import KaggleApi
from zipfile import ZipFile
import splitfolders
from glob import glob
import shutil
from pathlib import Path
import pandas as pd

def download(url: str, fname: str):
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get('content-length', 0))
    with open(fname, 'wb') as file, tqdm(
        desc=fname,
        total=total,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)

def create_model():
    '''Create and compile a deep neural net based on InceptionV3.'''
    pre_trained_model = InceptionV3(input_shape=(150, 150, 3), include_top=False, weights='weights.h5')
    
    unfreeze = False

    # Unfreeze all models after "mixed6"
    for layer in pre_trained_model.layers:
        if unfreeze:
            layer.trainable = True
        if layer.name == 'mixed6':
            unfreeze = True

    last_layer = pre_trained_model.get_layer('mixed7')
    last_output = last_layer.output

    # Flatten the output layer to 1 dimension
    x = layers.Flatten()(last_output)

    # Add a fully connected layer with 1,024 hidden units and ReLU activation
    x = layers.Dense(1024, activation='relu')(x)

    # Add a dropout rate of 0.2
    x = layers.Dropout(0.2)(x)

    # Add a final sigmoid layer for classification
    x = layers.Dense(1, activation='sigmoid')(x)

    model = Model(pre_trained_model.input, x)
    model.compile(optimizer=optimizers.SGD(learning_rate=0.00001, momentum=0.9), loss='binary_crossentropy', metrics=['accuracy'])

    return model    

def train():    
    # The following variables are the hyperparameters.
    batch_size = 20
    epochs = 50

    # Add our data-augmentation parameters to ImageDataGenerator
    train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    train_generator = train_datagen.flow_from_directory('dataset/train', target_size=(150, 150), batch_size=batch_size, class_mode='binary')

    # Note that the validation data should not be augmented!
    val_datagen = ImageDataGenerator(rescale=1./255)
    validation_generator = val_datagen.flow_from_directory('dataset/val', target_size=(150, 150), batch_size=batch_size, class_mode='binary')

    # Establish the model's topography.
    model = create_model()

    # Train the model on the normalized training set.
    model.fit(train_generator, steps_per_epoch=train_generator.samples // batch_size, epochs=epochs, validation_data=validation_generator, validation_steps=validation_generator.samples // batch_size)

    # Save the model for testing.
    model.save('model', save_format='h5')
    
    # Save the model for use in the frontend.
    shutil.rmtree('app/public/model')
    tfjs.converters.save_keras_model(model, 'app/public/model')

def test(filename: str):
    # Load the model.
    model: Model = models.load_model('model.h5') 

    # Load the test image.
    resized = np.array(Image.open(filename).resize((150, 150)))
    normalized = np.interp(resized, (resized.min(), resized.max()), (0, 1))
    batched = normalized[None,:,:]
    
    # Predict the class of the test image.
    pred = model.predict(batched)
    number_pred = round(pred[0][0])
    print('Cat') if number_pred == 0 else print('Dog')

def testAll():
    # Load the model.
    model: Model = models.load_model('model.h5') 

    # Load all test images.
    test_images = glob('dataset/test/*.jpg')
    
    data = {'id': [], 'label': []} 

    # Predict the class of each test image.
    for image in tqdm(test_images):
        resized = np.array(Image.open(image).resize((150, 150)))
        normalized = np.interp(resized, (resized.min(), resized.max()), (0, 1))
        batched = normalized[None,:,:]
        
        pred = model.predict(batched)
        number_pred = round(pred[0][0])

        # Add the image's id and predicted class to the data dictionary.
        data['id'].append(Path(image).stem)
        data['label'].append(number_pred)

    # Save the data dictionary to a csv file.
    df = pd.DataFrame(data)
    df.sort_values(by=['id'], inplace=True)
    df.to_csv('submission.csv', index=False)



if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    api = KaggleApi()
    api.authenticate()
    if not path.exists('dataset'):
        api.competition_download_files('dogs-vs-cats', path='dataset', quiet=False)
        with ZipFile('dataset/dogs-vs-cats.zip', 'r') as dataset_zip:
            dataset_zip.extractall('dataset')
        remove('dataset/dogs-vs-cats.zip')
        with ZipFile('dataset/test1.zip', 'r') as test_zip:
            test_zip.extractall('dataset')
            rename('dataset/test1', 'dataset/test')
        remove('dataset/test1.zip')
        with ZipFile('dataset/train.zip', 'r') as train_zip:
            train_zip.extractall('dataset')
        remove('dataset/train.zip')
        remove('dataset/sampleSubmission.csv')
        mkdir('dataset/train/cat')
        cats = glob('dataset/train/cat.*.jpg')
        for cat in cats:
            shutil.move(cat, 'dataset/train/cat')
        mkdir('dataset/train/dog')
        dogs = glob('dataset/train/dog.*.jpg')
        for dog in dogs:
            shutil.move(dog, 'dataset/train/dog')
        splitfolders.ratio('dataset/train', output='dataset/temp', seed=420, ratio=(.8, .2), move=True)
        shutil.rmtree('dataset/train')
        shutil.move('dataset/temp/train', 'dataset')
        shutil.move('dataset/temp/val', 'dataset')
        shutil.rmtree('dataset/temp')
    if not path.exists('weights.h5'):
        download('https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5', 'weights.h5')
    if not path.exists('model.h5'):
        train()
    if not path.exists('submission.csv'):
        testAll()
    else:
        filename = input('Enter the filename of the test image: ')
        test(filename)
