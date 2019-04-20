from keras import layers, models, optimizers
from keras import backend as K
from keras.layers import BatchNormalization, Activation
from keras import initializers
from keras import regularizers
import numpy as np

# DDPG (Deep Deterministic Policy Gradients) Critic Model
# Neural Network: (State,Action) -> Q-value
# The purpose is actually to learn Policy Gradients, 
# i.e. derivatives of Q-values with respect to Actions (needed for training Actor)


class CriticNetwork:

    """Critic (Value) Model."""

    def __init__(self, state_size, action_size):

        self.state_size = state_size
        self.action_size = action_size
        self.build_model()

    def build_model(self):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        # Define input layers

        w_init = initializers.RandomUniform(minval=-0.05, maxval=0.05)

        init_state = initializers.RandomUniform(minval=-0.07, maxval=0.07)

        states = layers.Input(shape=(self.state_size,), name='states')
        actions = layers.Input(shape=(self.action_size,), name='actions')

        net_states = layers.Dense(units=400)(states)
        net_states = BatchNormalization()(net_states)
        net_states = Activation('relu')(net_states)

        net_states = layers.Dense(units=200, kernel_initializer=w_init)(net_states)

        net_actions = layers.Dense(units=200)(actions)

        # Combine state and action pathways
        net = layers.Add()([net_states, net_actions])
        net = layers.Activation('relu')(net)

        # Add more layers to the combined network if needed

        # Add final output layer to prduce action values (Q values)
        Q_values = layers.Dense(units=1, kernel_initializer=init_state, name='q_values')(net)

        # Create Keras model
        self.model = models.Model(inputs=[states, actions], outputs=Q_values)

        # Define optimizer and compile model for training with built-in loss function
        optimizer = optimizers.Adam(lr=0.001)
        self.model.compile(optimizer=optimizer, loss='mse')

        # Compute action gradients (derivative of Q values w.r.t. to actions)
        action_gradients = K.gradients(Q_values, actions)

        # Define an additional function to fetch action gradients (to be used by actor model)
        self.get_action_gradients = K.function(
            inputs=[*self.model.input, K.learning_phase()],
            outputs=action_gradients)