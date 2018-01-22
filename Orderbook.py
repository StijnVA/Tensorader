import gdax
import os
import errno
import json
import time

directory = 'C:\Conyza\orderbookdata\data'

try:
    os.makedirs(directory)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

public_client = gdax.PublicClient()

counter = 0
while True:
    try:
        data = {}
        data['history'] = public_client.get_product_historic_rates('BTC-EUR', granularity=100)
        orderbook = public_client.get_product_order_book('BTC-EUR', level=2)
        data['orderbook'] = orderbook
        filename = directory + '/' + str(orderbook['sequence']) + '.json'
        with open(filename, mode= 'w+') as file:
            json.dump(data, file)
        counter += 1
        time.sleep(0.5)
        if counter % 1000 == 0 :
            print ('collected ' + str(counter) + ' files')
    except Exception as error:
        print(error)
