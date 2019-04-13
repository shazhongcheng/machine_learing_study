#mnist_lenet5_test.py
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import forward
import backward

TEST_INTERVAL_SECS = 5

def test(mnist):
    with tf.Graph().as_default() as g:
        x=tf.placeholder(tf.float32,[
            mnist.test.num_examples,
            forward.IMAGE_SIZE,
            forward.IMAGE_SIZE,
            forward.NUM_CHANNELS])
        y_=tf.placeholder(tf.float32,[None,forward.OUTPUT_NODE])
        y=forward.forward(x,False,None)
        
#         ema=tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
#         ema_restore=ema.variables_to_restore()
#         saver = tf.train.Saver(ema_restore)
        ema=tf.train.ExponentialMovingAverage(backward.MOVING_AVERAGE_DECAY)
        ema_restore = ema.variables_to_restore()
        saver=tf.train.Saver(ema_restore)
        
        correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
        accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        
        with tf.Session(config=config) as sess:
            ckpt = tf.train.get_checkpoint_state(backward.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess,ckpt.model_checkpoint_path)
                global_step=ckpt.model_checkpoint_path.split('/')[-1].split('.')[-1]
                reshape_x=np.reshape(mnist.test.images,(
                    mnist.test.num_examples,
                    forward.IMAGE_SIZE,
                    forward.IMAGE_SIZE,
                    forward.NUM_CHANNELS
                ))
                accuracy_score=sess.run(accuracy,feed_dict={x:reshape_x,y_:mnist.test.labels})
                print("After %s training step(s), test accuracy = %g"%(global_step,accuracy_score))
            else:
                print("No checkpoint file found")
                return
def main():
    mnist = input_data.read_data_sets("./MNIST_data/",one_hot=True)
    test(mnist)          
    
if __name__=="__main__":
    main()
        