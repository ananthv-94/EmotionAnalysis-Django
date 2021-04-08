from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import os
import pdb

import tensorflow as tf

graph = tf.get_default_graph()

# size of the image: 48*48 pixels
pic_size = 48
# number of possible label values
nb_classes = 7
# number of epochs to train the N
epochs = 50

cwd = os.getcwd()
base_path = os.path.join(cwd,"static")

from keras.models import model_from_json
def create_model( model_json_file, model_weights_file):
        # load model from JSON file
        with open(model_json_file, "r") as json_file:
            loaded_model_json = json_file.read()
            loaded_model = model_from_json(loaded_model_json)

        # load weights into the new model
        loaded_model.load_weights(model_weights_file)
        return loaded_model

cnn_model = create_model("model.json", "model_weights.h5")

 
class Train_data:
    def data_generation(self):
        # number of images to feed into the NN for every batch
        batch_size = 128
        pic_size = 48
        datagen_train = ImageDataGenerator()
        datagen_validation = ImageDataGenerator()
        

        train_generator = datagen_train.flow_from_directory(base_path + "/train",
                                                            target_size=(pic_size,pic_size),
                                                            color_mode="grayscale",
                                                            batch_size=batch_size,
                                                            class_mode='categorical',
                                                            shuffle=True)

        validation_generator = datagen_validation.flow_from_directory(base_path + "/validation",
                                                            target_size=(pic_size,pic_size),
                                                            color_mode="grayscale",
                                                            batch_size=batch_size,
                                                            class_mode='categorical',
                                                            shuffle=False)
        return train_generator, validation_generator

    def transfer_learn(self, train_generator, validation_generator):
        with graph.as_default():
            model = cnn_model
            model.compile(optimizer="Adam" , loss='categorical_crossentropy', metrics=['accuracy'])
            model.fit_generator(generator=train_generator,
                                            steps_per_epoch=train_generator.n//train_generator.batch_size,
                                            epochs=50,
                                            validation_data = validation_generator,
                                            validation_steps = validation_generator.n//validation_generator.batch_size
                                            )

train_data_obj = Train_data()




