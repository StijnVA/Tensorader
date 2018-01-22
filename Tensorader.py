import tensorflow as tf
import math
import numpy as np

lastTrades = []

def doAction(btc, euro, action, price):
    """
    print('Action: ', action)
    print('   Price: ', price)
    print('   €', euro, ' - ', btc)
    """
    #TransactionCost = 0.025
    TransactionCost = 0.05
    if action == 0: 
        #print('buying @ ', price)
        factor = 1
        buy = euro * factor / price
        btc += buy
        euro -= euro * factor
        btc -= TransactionCost * buy
    """
    if action == 1:
        #print('buying @ ', price)
        factor = 0.5
        buy = euro * factor / price
        btc += buy
        euro -= euro * factor
        euro -= 0.02 * buy
    if action == 2:
        #print('buying @ ', price)
        factor = 0.3
        buy = euro * factor / price
        btc += buy
        euro -= euro * factor
        euro -= 0.02 * buy
    #if action == 3:
        #print('holding on')
    if action == 4:
        #print('selling ', btc, ' BTC @ ', price)
        factor = 0.3
        sell = btc * factor * price
        euro += sell
        btc -= btc * factor
        euro -= 0.02 * sell
    if action == 5:
        #print('selling ', btc, ' BTC @ ', price)
        factor = 0.5        
        sell = btc * factor * price
        euro += sell
        btc -= btc * factor
        euro -= 0.02 * sell
    """
    if action == 2:
        #print('selling ', btc, ' BTC @ ', price)
        factor = 1        
        sell = btc * factor * price
        euro += sell
        btc -= btc * factor
        euro -= TransactionCost * sell
        
    euro = (math.floor(euro * 100)) / 100.0
    return btc, euro
    
def create_file_reader_ops(filename_queue):
    reader = tf.TextLineReader(skip_header_lines=0)
    record_defaults = [[0], [0.0], [0.0]]
    if len(lastTrades) > 0:
        lastTrades.pop(0)
    while len(lastTrades) < 50:
        _, csv_row = reader.read(filename_queue)
        timestamp, price, volume = tf.decode_csv(csv_row, record_defaults=record_defaults)
        #lastTrades.append([price, volume])
        lastTrades.append([price])
    return lastTrades

filename_queue = tf.train.string_input_producer(["..\..\data\.coinbaseEUR.csv"])
trade_reader = create_file_reader_ops(filename_queue)


NumberOfActions = 3

trades_placeholder = tf.placeholder(shape=[None, 50,1],dtype=tf.float32, name='trades')
wallet_placeholder = tf.placeholder(shape=[None, 2], dtype=tf.float32, name='wallet')
actions_placeholder = tf.placeholder(shape=[None], dtype=tf.int32)
rewards_placeholder = tf.placeholder(shape=[None], dtype=tf.float32)

#Y0 = tf.layers.dense(tick_placeholder. tf.reshape(trades_placeholder, [-1, 100]))

Y1 = tf.layers.dense(tf.reshape(trades_placeholder ,[-1,50]), 200, activation=tf.nn.sigmoid)
Y2 = tf.layers.dense(Y1, 150, activation=tf.nn.sigmoid)
Y3 = tf.layers.dense(Y2, 100, activation=tf.nn.sigmoid)
#Y4 = tf.layers.dense(Y3, 100, activation=tf.nn.sigmoid)
#Y5 = tf.layers.dense(Y4, 50, activation=tf.nn.sigmoid)
YN = tf.layers.dense(Y3, NumberOfActions, activation=tf.nn.softmax)

sample_op = tf.multinomial(logits=YN, num_samples=1)

#cross_entropies = tf.losses.softmax_cross_entropy(onehot_labels=tf.one_hot(actions_placeholder, 5), logits=YN) 
#loss = tf.reduce_sum(-rewards_placeholder * cross_entropies)
#loss = tf.sigmoid(rewards_placeholder * cross_entropies)
#optimizer = tf.train.RMSPropOptimizer(learning_rate=0.003, decay=0.99)

cross_entropies = tf.losses.softmax_cross_entropy(onehot_labels=tf.one_hot(actions_placeholder, NumberOfActions), logits=YN) 
#loss = -(tf.log(cross_entropies)*rewards_placeholder)
loss = tf.log(cross_entropies)*rewards_placeholder
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.003)
#optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
update = optimizer.minimize(loss)

train_op = optimizer.minimize(loss)

with tf.Session() as sess:
  # Start populating the filename queue.
  tf.global_variables_initializer().run()
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord)
  reward = -1
  i = 0
  ticks = 50
  last_rewards = []
  median = 0
  while median < 10:
    btc = 0.0
    euro = 100.0
    observations = []
    actions = []
    wallets = []
    interactions = [0] * NumberOfActions
    for tick in range(ticks):
        trade_data = sess.run(trade_reader)
        price = trade_data[-1][0]
        wallet = [euro, btc]
        sample_data = {trades_placeholder: [trade_data], wallet_placeholder: [wallet] }
        action = sess.run(sample_op, feed_dict=sample_data)
        interaction = action[0][0]
        observations.append(trade_data)
        actions.append(action[0][0])
        wallets.append(wallet)
        interactions[interaction] = interactions[interaction] + 1
        btc, euro = doAction(btc, euro, action[0][0], price)
    
    reward = (((euro + btc * price) / 100) - 1) * 100
    #reward = ((euro / 100) - 1) * 100
    #rewards.append(reward)
    rewards = [reward] * ticks
    last_rewards.append(reward)
    if i % 25 == 0 :
        median = np.median(last_rewards)
        print(i,': reward:', reward, '[median: ', median, '] (' , interactions,') €', euro, ' BTC ', btc)
        last_rewards.clear()
    train_data = { trades_placeholder: observations, wallet_placeholder: wallets,  actions_placeholder: actions, rewards_placeholder: rewards}
    sess.run(train_op, feed_dict=train_data)
    i += 1
  coord.request_stop()
  coord.join(threads)