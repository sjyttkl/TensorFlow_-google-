# -*- coding: utf-8 -*-

"""
==================================================
   File Name：     range_input_producer
   email:         songdongdong@jd.com
   Author :       songdongdong
   date：          2019/3/25
   Description :  
==================================================
"""
__author__ = 'songdongdong'

import tensorflow as tf
import codecs

BATCH_SIZE = 6
NUM_EXPOCHES = 5


def input_producer():
    array = codecs.open("test.txt","r",encoding="utf-8").readlines()
    # array = map(lambda line: line.strip(), array)
    array = [a.strip() for a in array]
    i = tf.train.range_input_producer(NUM_EXPOCHES, num_epochs=3, shuffle=False).dequeue()#0-4
    inputs = tf.slice(array, [i * BATCH_SIZE], [BATCH_SIZE])  #begin 和  size
    print(i)
    return inputs


class Inputs(object):
    def __init__(self):
        self.inputs = input_producer()


def main():
    inputs = Inputs()
    init = tf.group(tf.initialize_all_variables(),
                    tf.initialize_local_variables())
    sess = tf.Session()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    sess.run(init)
    try:
        index = 0
        while not coord.should_stop() and index < 10:
            datalines = sess.run(inputs.inputs)
            index += 1
            print("step: %d, batch data: %s" % (index, str(datalines)))
    except tf.errors.OutOfRangeError:
        print("Done traing:-------Epoch limit reached")
    except KeyboardInterrupt:
        print("keyboard interrput detected, stop training")
    finally:
        coord.request_stop()
    coord.join(threads)
    sess.close()
    del sess


if __name__ == "__main__":
    main()