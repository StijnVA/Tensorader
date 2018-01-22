import os.path
import numpy as np
import tensorflow as tf
import math
import time


#OBSERVATIONS_SIZE = 6400
WALLET_SIZE = 2
History_SIZE = 50 * 5
#History_SIZE = 5


class Network:
    def __init__(self, hidden_layer_size, learning_rate, checkpoints_dir, numberOfActions):
        self.learning_rate = learning_rate

        self.sess = tf.InteractiveSession()

        self.trainingcounter = 0

        self.wallet = tf.placeholder(tf.float32, [None, WALLET_SIZE])
        
        self.history = tf.placeholder(tf.float32, [None, History_SIZE])

        self.sampled_actions = tf.placeholder(tf.float32, [None, numberOfActions])
        
        self.advantage = tf.placeholder(tf.float32, [None, 1], name='advantage')

        x1 = tf.layers.dense(
             self.wallet,
             units=5,
             activation=tf.nn.relu,
             kernel_initializer=tf.contrib.layers.xavier_initializer())
        
        x2 = tf.layers.dense(
             self.history,
             units=500,
             activation=tf.nn.relu,
             kernel_initializer=tf.contrib.layers.xavier_initializer())

        x2a = tf.layers.dense(
             x2,
             units=300,
             activation=tf.nn.relu,
             kernel_initializer=tf.contrib.layers.xavier_initializer())

        x2b = tf.layers.dense(
             x2a,
             units=100,
             activation=tf.nn.relu,
             kernel_initializer=tf.contrib.layers.xavier_initializer())

        x2c = tf.layers.dense(
             x2b,
             units=5,
             activation=tf.nn.relu,
             kernel_initializer=tf.contrib.layers.xavier_initializer())

        h = tf.layers.dense(
            tf.concat([x1, x2c],1),
            units=100,
            activation=tf.nn.sigmoid,
            kernel_initializer=tf.contrib.layers.xavier_initializer())

        self.up_probability = tf.layers.dense(
            h,
            units=numberOfActions,
            activation=tf.nn.softmax,
            kernel_initializer=tf.contrib.layers.xavier_initializer())

        # Train based on the log probability of the sampled action.
        # 
        # The idea is to encourage actions taken in rounds where the agent won,
        # and discourage actions in rounds where the agent lost.
        # More specifically, we want to increase the log probability of winning
        # actions, and decrease the log probability of losing actions.
        #
        # Which direction to push the log probability in is controlled by
        # 'advantage', which is the reward for each action in each round.
        # Positive reward pushes the log probability of chosen action up;
        # negative reward pushes the log probability of the chosen action down.
        self.loss = - tf.losses.log_loss(
            labels=self.sampled_actions,
            predictions=self.up_probability,
            weights=-self.advantage,
            reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS
            )

        # optimizer = tf.train.AdamOptimizer(self.learning_rate)
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        
        self.train_op = optimizer.minimize(self.loss)

        
        tf.global_variables_initializer().run()

        self.saver = tf.train.Saver()
        self.checkpoint_file = os.path.join(checkpoints_dir,
                                            'policy_network.ckpt')


        summary_loss = tf.summary.scalar("loss", self.loss)
        stats_advantage = tf.reduce_mean(self.advantage)
        summary_reward = tf.summary.scalar("reward", stats_advantage)

        self.summaries = tf.summary.merge([summary_loss, summary_reward])

        datetime = str(math.trunc(time.time()))
        self.summarywriter = tf.summary.FileWriter("log/" + datetime + "-training") 

    def load_checkpoint(self):
        print("Loading checkpoint...")
        self.saver.restore(self.sess, self.checkpoint_file)

    def save_checkpoint(self):
        print("Saving checkpoint...")
        self.saver.save(self.sess, self.checkpoint_file)

    def forward_pass(self, wallet, history):
        up_probability = self.sess.run(
            self.up_probability,
            feed_dict={
                self.wallet: wallet.reshape([-1, 2]),
                self.history: history.reshape([-1, 250])})
        return up_probability

    def train(self, state_action_reward_tuples):
        self.trainingcounter += 1
        print("Training with %d (state, action, reward) tuples" %
              len(state_action_reward_tuples))

        wallet, history, actions, rewards = zip(*state_action_reward_tuples)
        wallet = np.vstack(wallet)
        history = np.vstack(history)
        actions = np.vstack(actions)
        rewards = np.vstack(rewards)

        feed_dict = {
            self.wallet: wallet.reshape([-1,2]),
            self.history: history.reshape([-1,250]),
            self.sampled_actions: actions.reshape([-1,3]),
            self.advantage: rewards.reshape([-1,1])
        }
        _, smm = self.sess.run([self.train_op, self.summaries], feed_dict)
        self.summarywriter.add_summary(smm, self.trainingcounter)
