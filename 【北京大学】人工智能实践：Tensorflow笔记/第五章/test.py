# mnist_test.py
# 进行测试
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import forward
import backward
TEST_INTERVAL_SECS = 5

# 定义TensorFlow配置
config = tf.ConfigProto()
 
# 配置GPU内存分配方式，按需增长，很关键
config.gpu_options.allow_growth = True
 
# 配置可使用的显存比例
config.gpu_options.per_process_gpu_memory_fraction = 0.1


def test(mnist):
    with tf.Graph().as_default() as g:
        x=tf.placeholder(tf.float32,[None,forward.INPUT_NODE])
        y_=tf.placeholder(tf.float32,[None,forward.OUTPUT_NODE])
        y=forward.forward(x,None)

        ema=tf.train.ExponentialMovingAverage(backward.MOVING_AVERAGE_DECAY)
        ema_restore = ema.variables_to_restore()
        saver=tf.train.Saver(ema_restore)
        
        correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
        accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        

        with tf.Session(config = config) as sess:
            ckpt = tf.train.get_checkpoint_state(backward.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess,ckpt.model_checkpoint_path)
                global_step=ckpt.model_checkpoint_path.split('/')[-1].split('.')[-1]
                accuracy_score=sess.run(accuracy,feed_dict={x:mnist.test.images,y_:mnist.test.labels})
                print('After %s traing step(s),test accuracy=%g'%(global_step,accuracy_score))
            else:
                print('No checkpoint file found')
                return

def main():
	mnist=input_data.read_data_sets("MNIST_data/",one_hot=True)
	test(mnist)

if __name__=='__main__':
	main()