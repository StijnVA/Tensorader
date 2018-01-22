import tensorflow as tf
import math
import numpy as np
import os
from random import randint
import json
import time
from policy_network import Network



directory = 'C:\\Conyza\\orderbookdata\\data\\'
#sample_size = 10000
#sample_size = 100
sample_size = 20
#batch_size = 25
#batch_size = 100
batch_size = 5

history_size = 50
orderbook_size = 20

ACTION_NONE = 0
ACTION_SELL = 1
ACTION_BUY = 2
NumberOfActions = 3
WRONG_ACTION_REWARD = 0
#TransactionCost = 0.05
TransactionCost = -0.10

def getTrainingData():
    files = os.listdir(directory)
    start = randint(0, len(files) - sample_size)
    batchfiles = files[start:start+sample_size]
    for filename in batchfiles :
        try:
            obj = json.load(open(directory + filename))
            history = obj['history']
            bids = obj['orderbook']['bids']
            asks = obj['orderbook']['asks'] 
            for h in history :
               del h[0]
            for b in bids :
                for i, item in enumerate(b) :
                    b[i] = float(item)
            for a in asks :
                for i, item in enumerate(a) :
                    a[i] = float(item)
            while len(history) < history_size :
                history.append([0,0,0,0,0])
            while len(bids) < orderbook_size :
                bids.append([0,0,0])
            while len(asks) < orderbook_size :
                asks.append([0,0,0])
            yield history[:history_size], bids[:orderbook_size], asks[:orderbook_size]
        except Exception as error:
            print(error, ' in file ', filename)
            os.remove(directory + filename)
            yield np.zeros([history_size,5]), np.zeros([orderbook_size,3]), np.zeros([orderbook_size,3])

def doAction(btc, euro, action, bids, asks):
    #TransactionCost = 0.025
    price = 0
    
    if action == ACTION_BUY: 
        price = asks[0][0]
        if price != 0 :       
        #print('buying @ ', price)
            factor = 1
            buy = euro * factor / price
            btc += buy
            euro -= euro * factor
            btc -= TransactionCost * buy
    if action == ACTION_NONE :
        price = (bids[0][0] + asks[0][0]) / 2
    if action == ACTION_SELL:
        #print('selling ', btc, ' BTC @ ', price)
        price = bids[0][0]
        factor = 1        
        sell = btc * factor * price
        euro += sell
        btc -= btc * factor
        euro -= TransactionCost * sell
        
    euro = (math.floor(euro * 100)) / 100.0
    return btc, euro, price

network = Network(50, 0.03, numberOfActions=3,  checkpoints_dir='checkpoints')

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    reward = 0
    i = 0
    ticks = 0
    last_rewards = []
    median = 0
   

    batch_state_action_reward_tuples = []

    while median < 10:
        i += 1
        btc = 0.0
        euro = 100.0
        lastposition = 0
        for history, bids, asks in getTrainingData():
            wallet = [euro, btc]
           
            wallet = np.array(wallet).ravel()
            history = np.array(history) #todo: 50x5 instead of 250 flat
            
            propabilitieMatrix = network.forward_pass(wallet, history)
            print('Propabilitie Matrix', len(propabilitieMatrix), 'x', len(propabilitieMatrix[0]))
            propabilities = propabilitieMatrix[0]

            #print(propabilities, sample_data)
            if np.random.uniform() < propabilities[0]:
                action = ACTION_BUY
            elif np.random.uniform() < propabilities[1]:
                action = ACTION_SELL
            else:
                action = ACTION_NONE
            
            btc_new, euro_new, price = doAction(btc, euro, action, bids, asks)
            
            ticks += 1
            if action == ACTION_NONE :
                reward = 'none'
            elif action == ACTION_BUY :
                if euro == 0 :
                    reward = WRONG_ACTION_REWARD
                else :
                    reward = 'buy'
                    lastposition = euro
            elif action == ACTION_SELL : 
                if btc == 0 :
                   reward = WRONG_ACTION_REWARD
                else :
                    reward = euro_new / lastposition
                    # reward = (euro_new / lastposition) -1
                    # reward = (euro_new - lastposition) / 100
                    wallet2, history2, actions2, rewards2 = zip(*batch_state_action_reward_tuples)
                    # rewards2 = [reward if (x == 'buy') else reward if(x == 'none') else x for x in rewards2]
                    rewards2 = [reward if (x == 'buy') else 0 if(x == 'none') else x for x in rewards2]
                    batch_state_action_reward_tuples = list(zip(wallet2, history2, actions2, rewards2))

            tup = (wallet, history, [1 if action == ACTION_BUY else 0, 1 if action == ACTION_SELL else 0, 1 if action == ACTION_NONE else 0], reward)
            batch_state_action_reward_tuples.append(tup)
                    
            btc = btc_new
            euro = euro_new
   
        if i % batch_size == 0 :
            
            wallet, history, actions, rewards = zip(*batch_state_action_reward_tuples)

            rewards = [1 if (x == 'buy') else -1 if (x == 'none') else x for x in rewards]
            # rewards = [(r-1) * 100 for r in rewards]

            print(np.average(rewards))
            print(rewards)
            print(actions)
            batch_state_action_reward_tuples = list(zip(wallet, history, actions, rewards))
            network.train(batch_state_action_reward_tuples)
            batch_state_action_reward_tuples = []

            tf.train.Saver().save(sess, './savepoints/AlphaA-1.ckpt')

    coord.request_stop()
    coord.join(threads)