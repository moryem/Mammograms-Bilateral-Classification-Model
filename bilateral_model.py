# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 20:56:50 2018

@author: Mor
"""

from keras.models import Model 
from keras.layers import Dropout, BatchNormalization, LeakyReLU, Flatten, Dense
from keras.layers import Conv2D, MaxPooling2D, Input, Lambda
from keras import optimizers
from keras.callbacks import EarlyStopping #, LearningRateScheduler
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import json
import os
import keras.backend as K


class bilateral_net:
    
    def __init__(self, dp, bs, ep, lr, y_train, y_test, right_train, right_test,
                 left_train, left_test, plt_loss):
# =============================================================================
#       Initialize parameters
# =============================================================================
        self.dp = dp                        # dropout probability
        self.bs = bs                        # batch size
        self.ep = ep                        # number of epochs
        self.lr = lr                        # learning rate
        self.y_train = y_train              # train targets
        self.y_test = y_test                # test targets
        self.right_train = right_train      # right net train data
        self.right_test = right_test        # right net test data
        self.left_train = left_train        # left net train data
        self.left_test = left_test          # left net test data
        self.plt_loss = plt_loss            # boolean - plot loss or not
        self.inp_shape = (right_train.shape[1],right_train.shape[2],right_train.shape[3])   # input shape
       
    def twin(self):
# =============================================================================
#       Model architecture        
# =============================================================================
        
        inp = Input(self.inp_shape)
        
        # conv and pooling 1
        x = Conv2D(16, (3, 3), strides=(1, 1), padding='same', input_shape=self.inp_shape)(inp)
        x = LeakyReLU(alpha=0.3)(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(self.dp)(x)

        # conv 2
        x = Conv2D(32, (3, 3), strides=(1, 1), padding='same')(x)
        x = LeakyReLU(alpha=0.3)(x)
        x = Conv2D(32, (3, 3), strides=(1, 1), padding='same')(x)
        x = LeakyReLU(alpha=0.3)(x)
        x = BatchNormalization()(x)

        # conv and pooling 3
        x = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(x)
        x = LeakyReLU(alpha=0.3)(x)
        x = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(x)
        x = LeakyReLU(alpha=0.3)(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(self.dp)(x)

        # conv 4
        x = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(x)
        x = LeakyReLU(alpha=0.3)(x)
        x = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(x)
        x = LeakyReLU(alpha=0.3)(x)
        x = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(x)
        x = LeakyReLU(alpha=0.3)(x)
        x = BatchNormalization()(x)

       # conv and pooling 5
        x = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(x)
        x = LeakyReLU(alpha=0.3)(x)
        x = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(x)
        x = LeakyReLU(alpha=0.3)(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(self.dp)(x)

        # conv 6
        x = Conv2D(32, (3, 3), strides=(1, 1), padding='same')(x)
        x = LeakyReLU(alpha=0.3)(x)
        x = Conv2D(32, (3, 3), strides=(1, 1), padding='same')(x)
        x = LeakyReLU(alpha=0.3)(x)
        x = BatchNormalization()(x)
        
        # FC layer
        x = Flatten()(x)
        x = Dense(128)(x)
        out = Dropout(self.dp)(x)
        
        # define the twin network
        self.twin_net = Model(inp,out)

    def net(self):
# =============================================================================
#       merged twin networks                 
# =============================================================================

        # Use the same model - weight sharing
        self.twin()

        in_l = Input(shape=self.inp_shape, name='left_input')
        out_l = self.twin_net(in_l)  

        in_r = Input(shape=self.inp_shape, name='right_input')
        out_r = self.twin_net(in_r)
               
        # Merge the outputs with absolute distance
#        abs_dif = Lambda(lambda x: K.sum(K.abs(x[0]-x[1]), axis=1))
        abs_dif = Lambda(lambda x: K.abs(x[0]-x[1]))
        dist = abs_dif([out_r, out_l])
        out = Dense(1, activation='sigmoid')(dist)
#        out = Activation('sigmoid')(dist)

        net = Model([in_r, in_l], out)

        adam = optimizers.adam(lr=self.lr, decay=self.lr/self.ep)
        net.compile(loss=self.contrastive_loss, optimizer=adam, metrics=['accuracy'])
        # Learning schedule and early stopping callbacks
        # lrate = LearningRateScheduler(self.step_decay)        
        monitor = EarlyStopping(monitor='val_loss', min_delta=1e-5,
                                patience=10, verbose=1, mode='auto')
        callbacks = [monitor]
        
        return net, callbacks
    
#     def step_decay(self, epoch):
# # =============================================================================
# #         Learning rate schedule
# # =============================================================================
       
#         initial_lrate = self.lr
#         drop = 0.5
#         epochs_drop = 10.0
    	
#         lr = initial_lrate * np.powor(drop, np.floor((1+epoch)/epochs_drop))
        
#         return lr

    def contrastive_loss(self, y_train, y_pred):
# =============================================================================
#       Contrastive loss function        
# =============================================================================

        margin = 1
        sqaure_pred = K.square(y_pred)
        margin_square = K.square(K.maximum(margin - y_pred, 0))
        loss = K.mean((1 - y_train) * sqaure_pred + y_train * margin_square)

        return loss
    

    def data_augment(self, set_name):
# =============================================================================
#       Perform data augmentation to enlarge the train set       
# =============================================================================
        
        # Define the image transformations here
        datagen = ImageDataGenerator(validation_split=0.3,
                                     brightness_range = (0, 0.5),
                                     width_shift_range = 0.1,
                                     height_shift_range = 0.1,
                                     rotation_range = 40)
 
        # if train set:
        if (set_name=='Train'):
            # create two same generators for train right and left inputs
            gen_right = datagen.flow(self.right_train, self.y_train, batch_size=self.bs, seed=222)
            gen_left = datagen.flow(self.left_train, self.y_train, batch_size=self.bs, seed=222)
        elif (set_name=='Val'):
            # create two same generators for validation right and left inputs
            gen_right = datagen.flow(self.right_train, self.y_train, batch_size=self.bs, 
                                     subset='validation', seed=222)
            gen_left = datagen.flow(self.left_train, self.y_train, batch_size=self.bs, 
                                    subset='validation', seed=222)           
 
        while True:
            gen_right_i = gen_right.next()
            gen_left_i = gen_left.next()
            yield [gen_right_i[0], gen_left_i[0]], gen_right_i[1]

    
    def train(self):
# =============================================================================
#       Train the model
# =============================================================================
        
        self.siam_model, callbacks = self.net()
        class_weight = {0:1., 1:2.}

        # train with augmentation        
        train_generator = self.data_augment(set_name='Train')
        val_generator = self.data_augment(set_name='Val')
        self.history = self.siam_model.fit_generator(train_generator, validation_data=val_generator,
                                                    steps_per_epoch=len((self.right_train)/self.bs)*2,
                                                    validation_steps=len((self.right_train)/self.bs)*2,                                                
                                                    epochs=self.ep, verbose=1,callbacks=callbacks, 
                                                    class_weight=class_weight)

#        # train without augmentation
#        self.history = self.siam_model.fit(x=[self.right_train, self.left_train], y=self.y_train,
#                                          validation_split=0.3, callbacks=callbacks, batch_size=self.bs,
#                                          epochs=self.ep, verbose=1, class_weight=class_weight)        
        
        
        # plot loss                                
        if self.plt_loss:
            self.plot_history_loss()
                    
        #save model and twin model
        self.save_model(self.siam_model, self.history.history, 'siam')
        self.save_model(self.twin_net, [], 'twin')
                        
    def test(self):
# =============================================================================
#         Test the model performance
# =============================================================================

        # Predict models performance on test set
        if os.path.isfile('./model_siam.json'):         
            self.loaded_siam = self.load_model('model_siam.json','weights_siam.h5')
            self.y_pred = self.loaded_siam.predict(x=[self.right_test, self.left_test],
                                                   batch_size=self.bs, verbose=1)

             # plot loss                                
            if self.plt_loss:
                self.plot_history_loss()
                
        else:
            print('Train a model first!')
             
    def plot_history_loss(self):
# =============================================================================
#       Plot loss history        
# =============================================================================

        try:
            with open('history_siam.json') as history: 
                hist = json.load(history)
        except:
            print('Creating new loss graph')
            hist = self.history.history
                    
        plt.plot(hist['loss'])
        plt.plot(hist['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig('loss')
        plt.show(); plt.close()
        
        
    def save_model(self, model, history, name):
# =============================================================================
#       Save model to disk
# =============================================================================
        
        # serialize model to JSON
        model_json = model.to_json()
        with open('model_' + name + '.json', 'w') as json_file:
            json_file.write(model_json)
        # serialize history to JSON
        if name == 'siam':
            with open('history_' + name + '.json', 'w') as f:
                json.dump(history, f)       
        # serialize weights to HDF5
        model.save_weights('weights_' + name + '.h5')
        print('Saved model to disk')
       
    def load_model(self, model_name, weights_name):
# =============================================================================
#       Load model from disk
# =============================================================================
        
        # load json and create model
        json_file = open(model_name, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights(weights_name)
        print('Loaded model from disk')
        
        return loaded_model     
        
