import time
import mnist_lenet5_forward
import  mnist_lenet5_backward
import  tensorflow as tf
from tensorflow.examples.tutorials.mnist import  input_data

TEST_INTERVAL_SECS=5
import  numpy as np

def test(mnist):
    with tf.Graph().as_default() as g:
        #由于所有的图结构都是由相同的模型定义的，因此不用重复加载相同的图结构
        #而是不用同一个模型恢复相同的图解钩，每次只是加载不同的变量值而已
        #卷积神经网络的占位需要指定个数，因为conv2d函数需要这个个树参数，而全连接网络不用指定
        x = tf.placeholder(tf.float32, [
            mnist.test.num_examples,
            mnist_lenet5_forward.IMAGE_SIZE,
            mnist_lenet5_forward.IMAGE_SIZE,
            mnist_lenet5_forward.NUM_CHANNELS
        ])
        y_ = tf.placeholder(tf.float32, [None, mnist_lenet5_forward.OUT_NODE])
        y = mnist_lenet5_forward.forward(x,False, None)

        ema=tf.train.ExponentialMovingAverage(mnist_lenet5_backward.MOVING_AVERAGE_DECAY)
        ema_restore=ema.variables_to_restore()
        saver=tf.train.Saver(ema_restore)
        #定义加载变量，不修改加载原始参数，修改了加载滑动平均值，里面的参数不影响，因为滑动平均值早就计算好了

        correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
        accuracy=tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


        with tf.Session() as sess:
            ckpt=tf.train.get_checkpoint_state(mnist_lenet5_backward.MODER_SAVE_PATH)
            print(ckpt)
            if ckpt and ckpt.all_model_checkpoint_paths:
                for path in ckpt.all_model_checkpoint_paths:
                    saver.restore(sess,path)
                    global_step=path.split('/')[-1].split('-')[-1]
                    reshape_x = np.reshape(mnist.test.images,
                                            (mnist.test.num_examples,
                                             mnist_lenet5_forward.IMAGE_SIZE,
                                             mnist_lenet5_forward.IMAGE_SIZE,
                                             mnist_lenet5_forward.NUM_CHANNELS
                                             ))
                    accuracy_scare=sess.run(accuracy,feed_dict={x:reshape_x,y_:mnist.test.labels})
                    print(global_step,accuracy_scare)
            else:
                print('no')
                return
        time.sleep(TEST_INTERVAL_SECS)
def main():
    mnist=input_data.read_data_sets('./data/',one_hot=True)
    test(mnist)

if __name__=='__main__':
    main()




