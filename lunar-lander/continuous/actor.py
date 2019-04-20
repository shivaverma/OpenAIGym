from keras import layers, models, optimizers
from keras import backend as K
from keras.layers import BatchNormalization, Activation
from keras import initializers
from keras import regularizers
import numpy as np

class ActorNetwork:

    def __init__(self, state_size, action_size, action_low, action_high):

        self.state_size = state_size
        self.action_size = action_size
        self.action_low = action_low
        self.action_high = action_high
        self.action_range = self.action_high - self.action_low
        self.build_model()

    def build_model(self):
        """Build an actor (policy) network that maps states -> actions."""
        # Define input layer (states)

        w_init = initializers.RandomUniform(minval=-0.05, maxval=0.05)

        init_state = initializers.RandomUniform(minval=-0.07, maxval=0.07)

        states = layers.Input(shape=(self.state_size,), name='states')

        net = layers.Dense(units=400, kernel_initializer=init_state)(states)
        net = BatchNormalization()(net)
        net = Activation('relu')(net)

        net = layers.Dense(units=200, kernel_initializer=w_init)(net)
        net = BatchNormalization()(net)
        net = Activation('relu')(net)

        actions = layers.Dense(units=self.action_size, kernel_initializer=init_state, activation='tanh')(net)

        # Scale [0, 1] output for each action dimension to proper range
        # actions = layers.Lambda(lambda x: (x * self.action_range) + self.action_low,
        #    name='actions')(raw_actions)

        # Create Keras model
        self.model = models.Model(inputs=states, outputs=actions)

        # print(self.model.layers, '\n')
        # print(self.model.layers[-1].get_weights()[0])

        # Define loss function using action value (Q value) gradients
        action_gradients = layers.Input(shape=(self.action_size,))
        loss = K.mean(-action_gradients * actions)

        # Incorporate any additional losses here (e.g. from regularizers)

        # Define optimizer and training function
        optimizer = optimizers.Adam(lr=0.0001)
        updates_op = optimizer.get_updates(params=self.model.trainable_weights, loss=loss)
        self.train_fn = K.function(
            inputs=[self.model.input, action_gradients, K.learning_phase()],
            outputs=[],
            updates=updates_op)