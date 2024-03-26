import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
import tensorflow_probability as tfp
import os, numpy as np
import tensorflow_models as tfm
# Set logging level to suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any other level you prefer

tf.get_logger().setLevel('ERROR')

# def _build_DQN(num_actions, num_layers, width, state_dim):
#     model = Sequential([
#         # layers.Input(shape=(None, state_dim)),  # Define input shape
#         layers.Dense(width),
#         layers.Embedding(input_dim=state_dim, output_dim=width),
#         tfm.vision.layers.PositionalEncoding(),
#         layers.Reshape((-1, state_dim, width)),
#         tfm.nlp.layers.TransformerEncoderBlock(
#             inner_dim=state_dim,
#             inner_activation='relu',
#             num_attention_heads=5,
#             intermediate_size=width,
#         ),
#         layers.Flatten(),  # Flatten output before the final dense layer
#         layers.Dense(width)
#     ])
#     return model

class Base(Model):
    def __init__(self, state_dim, n_actions, num_layers, width, model_path, name="critic"):
        super(Base, self).__init__()
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.num_layers = num_layers
        self.width = width
        self.model_name = name
        self.model_path = model_path

        # Create the encoder with desired parameters
        encoder = tfm.nlp.layers.TransformerEncoderBlock(
            # hidden_size=self.width,  # Corrected from d_model
            inner_dim=self.state_dim,
            inner_activation='relu',
            # num_layers=num_layers,
            num_attention_heads=self.n_actions,  # Corrected from num_heads
            intermediate_size=self.n_actions,  # Typical value
            # dropout_rate=0.1,
            # key_dim=None
        )

        inputs = layers.Input(shape=(None,))  # Define input shape
        x = layers.Dense(self.width)
        x = layers.Embedding(input_dim=self.state_dim, output_dim=self.n_actions)(inputs)
        x = tfm.vision.layers.PositionalEncoding()(x)
        # x = layers.Flatten()(x[0])
        x = tf.reshape(x[0], (-1, self.state_dim, self.n_actions))
        outputs = encoder(x)  # Use TransformerEncoder here
        # outputs = tf.reshape(outputs,(self.batch_size,-1))
        outputs = layers.Flatten()(outputs), 
        # outputs = layers.Dense(width)(outputs)
        self.model = Model(inputs=inputs, outputs=outputs)
        # self.model = _build_DQN(n_actions, num_layers, width, state_dim )
        # Optimizer and loss
        self.optimizer = Adam()
        self.loss = MeanSquaredError()

# class Base(Model):
#     def __init__(self, n_actions, num_layers, width, model_path, name=" critic"):
#         super(Base, self).__init__()
#         self.n_actions = n_actions
#         self.num_layers = num_layers
#         self.width = width
#         self.model_name = name
#         self.model_path = model_path
#         self.model = Sequential()
#         for _ in range(num_layers):
#             self.model.add(Dense(self.width, activation="relu"))
#         self.optimizer = Adam()
#         self.loss = MeanSquaredError()

class CriticNetwork(Base):
    def __init__(self, state_dim, n_actions, num_layers, width, model_path, name):
        super(CriticNetwork, self).__init__(state_dim,n_actions, num_layers, width, model_path, name)
        self.q = Dense(1, activation=None)

    def call(self, state, action, batch_size):
        print('state shape:', state.shape)
        print('action shape:', action.shape)
        state_action = tf.concat([state, action], axis=1)
        print('state_action shape:', state_action.shape)
        act_val = self.model(state_action)
        return self.q(tf.reshape(act_val,(batch_size,-1)))

class ValueNetwork(Base):
    def __init__(self, state_dim, action_dim, num_layers, width,model_path, name ):
        super(ValueNetwork, self).__init__(state_dim, action_dim, num_layers, width,model_path, name)
        self.v = Dense(1, activation=None)

    def call(self, state, batch_size):
        state_val = tf.reshape(self.model(state),(batch_size,-1))
        # print(state_val.shape)
        return self.v(state_val)

class ActorNetwork(Base):
    def __init__(self, state_dim, n_actions, num_layers, width, model_path, max_action, name):
        super(ActorNetwork, self).__init__(state_dim,n_actions, num_layers, width, model_path, name)
        self.max_action = max_action
        self.noise = 1e-6
        self.mu = Dense(self.n_actions, activation=None)
        self.sigma = Dense(self.n_actions, activation=None)

    def call(self, state, batch_size):
        print('state shape', state.shape)
        prob = tf.convert_to_tensor(self.model(state))
        # prob = tf.reshape(self.model(state),(batch_size,-1))
        print('prob shape', prob.shape)
        prob = layers.Flatten()(prob)
        print('Flattened prob shape', prob.shape)
        prob = tf.reshape(prob, (batch_size,-1))
        print('Reshaped Flattened prob shape', prob.shape)
        mu = self.mu(prob)
        print('mu shape', mu.shape)
        sigma = self.sigma(prob)
        print('sigma shape', sigma.shape)
        sigma = tf.clip_by_value(sigma, self.noise, 1)
        return mu, sigma

    def sample_normal(self, state, batch_size):
        mu, sigma = self.call(state, batch_size)
        probs = tfp.distributions.Normal(mu, sigma)
        actions = probs.sample()
        # print('action shape',actions.shape)
        action = tf.math.tanh(actions) * self.max_action
        # print('action - ',action)
        log_probs = probs.log_prob(actions)
        action = tf.clip_by_value(action, 0, self.max_action)
        # action = tf.minimum(tf.maximum(action, 0), self.max_action - 1)
        # print('clipped action',action)
        log_probs -= tf.math.log(1 - tf.math.pow(action, 2) + self.noise)
        log_probs = tf.math.reduce_sum(log_probs, axis=1, keepdims=True)
        print('log probs shape',log_probs.shape)
        entropy =  tf.reduce_sum( probs.entropy(), axis=1, keepdims=True)
        return action, log_probs, entropy

class Agent:
    def __init__(self, state_dim, action_dim, max_action, learning_rate, epsilon, gamma, tau, decay_rate, model_path, num_layers=5, width=32, reward_scale=1):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.tau = tau
        self.decay_rate = decay_rate
        self.num_layers = num_layers 
        self.width = width 
        self.max_action = max_action
        self.model_path = model_path
        self.reward_scale = reward_scale
        self.actor = ActorNetwork(state_dim, action_dim, self.num_layers, self.width, self.model_path, self.max_action, name='Actor')
        self.critic_1 = CriticNetwork(state_dim, action_dim, self.num_layers, self.width, self.model_path, name='Critic-1')
        self.critic_2 = CriticNetwork(state_dim, action_dim, self.num_layers, self.width, self.model_path, name='Critic-2')
        self.value = ValueNetwork(state_dim, action_dim, self.num_layers, self.width,self.model_path, name='Value')
        self.target_value = ValueNetwork(state_dim, action_dim, self.num_layers, self.width,self.model_path, name='Target Value')
        
        self.actor.compile(optimizer = Adam(learning_rate))
        self.critic_1.compile(optimizer = Adam(learning_rate))
        self.critic_2.compile(optimizer = Adam(learning_rate))
        self.value.compile(optimizer = Adam(learning_rate))
        self.target_value.compile(optimizer = Adam(learning_rate))
        
        # self.update_network_params(tau=1)
    def act(self, state, batch_size=1):
        state = tf.convert_to_tensor([state])
        # print('state shape',state.shape)
        actions,_, _ = self.actor.sample_normal(state,batch_size)
        print(actions.shape)
        # print(np.argmax(actions))
        return actions
        
    def update_network_params(self,tau=None):
        if tau is None:
            tau = self.tau
        wts = []
        targets  = self.target_value.weights
        for i, wt in enumerate(self.value.weights):
            wts.append(wt*tau + targets[i]*(1-tau))
        self.target_value.set_weights(wts)
    
    def learn(self, experience, batch_size):
        states,actions,rewards,next_states, dones = tuple(map(lambda x: tf.convert_to_tensor(x,dtype=tf.float32),experience))
        
        with tf.GradientTape() as tape:
            val = tf.squeeze(self.value(states, batch_size),1)
            next_val = tf.squeeze(self.target_value(next_states,batch_size),1)

            curr_pol_actions, log_probs, _ = self.actor.sample_normal(states, batch_size)
            log_probs = tf.squeeze(log_probs,1)
            q1_new_pol = self.critic_1(states,curr_pol_actions, batch_size)
            q2_new_pol = self.critic_2(states,curr_pol_actions, batch_size)
            critic_val = tf.squeeze(tf.math.minimum(q1_new_pol,q2_new_pol),1)
            
            val_target = critic_val - log_probs
            val_loss = 0.5* MeanSquaredError()(val, val_target)
        val_nwk_grad = tape.gradient(val_loss, self.value.trainable_variables)
        self.value.optimizer.apply_gradients(zip(val_nwk_grad, self.value.trainable_variables))
        
        with tf.GradientTape() as tape:
            new_pol_actions, log_probs, entropy = self.actor.sample_normal(states, batch_size)
            log_probs = tf.squeeze(log_probs,1)
            q1_new_pol = self.critic_1(states,new_pol_actions, batch_size)
            q2_new_pol = self.critic_2(states,new_pol_actions, batch_size)
            critic_val = tf.squeeze(tf.math.minimum(q1_new_pol,q2_new_pol),1)
            # actor_loss = - log_probs * critic_val
            actor_loss = -log_probs * critic_val + self.epsilon * tf.reduce_mean(entropy)
            self.epsilon *= self.decay_rate
        actor_nwk_grad = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(actor_nwk_grad, self.actor.trainable_variables))

        with tf.GradientTape(persistent=True) as tape:
            q_hat = self.reward_scale * rewards + self.gamma* next_val * (1 - dones)
            q1_old_pol = tf.squeeze(self.critic_1(states,actions, batch_size),1)
            q2_old_pol = tf.squeeze(self.critic_2(states,actions, batch_size),1)
            critic_1_loss = 0.5 * MeanSquaredError()(q1_old_pol, q_hat)
            critic_2_loss = 0.5 * MeanSquaredError()(q2_old_pol, q_hat)
        critic_1_nwk_grad = tape.gradient(critic_1_loss, self.critic_1.trainable_variables)
        critic_2_nwk_grad = tape.gradient(critic_2_loss, self.critic_2.trainable_variables)
        self.critic_1.optimizer.apply_gradients(zip(critic_1_nwk_grad, self.critic_1.trainable_variables))
        self.critic_2.optimizer.apply_gradients(zip(critic_2_nwk_grad, self.critic_2.trainable_variables))
        
        self.update_network_params()
        
    def save_models(self, ts):
        self.actor.save_weights(self.model_path+f'{ts}-'+self.actor.model_name+'.chkpt')
        self.critic_1.save_weights(self.model_path+f'{ts}-'+self.critic_1.model_name+'.chkpt')
        self.critic_2.save_weights(self.model_path+f'{ts}-'+self.critic_2.model_name+'.chkpt')
        self.value.save_weights(self.model_path+f'{ts}-'+self.value.model_name+'.chkpt')
        self.target_value.save_weights(self.model_path+f'{ts}-'+self.target_value.model_name+'.chkpt')

    def load_models(self, ts):
        self.actor.load_weights(self.model_path+f'{ts}-'+self.actor.model_name+'.chkpt')
        self.critic_1.load_weights(self.model_path+f'{ts}-'+self.critic_1.model_name+'.chkpt')
        self.critic_2.load_weights(self.model_path+f'{ts}-'+self.critic_2.model_name+'.chkpt')
        self.value.load_weights(self.model_path+f'{ts}-'+self.value.model_name+'.chkpt')
        self.target_value.load_weights(self.model_path+f'{ts}-'+self.target_value.model_name+'.chkpt')
        