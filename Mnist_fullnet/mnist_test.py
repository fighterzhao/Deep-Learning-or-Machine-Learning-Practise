import time
import mnist_forward
import  mnist_backward
import  tensorflow as tf
from tensorflow.examples.tutorials.mnist import  input_data

TEST_INTERVAL_SECS=5

def test(mnist):
    with tf.Graph().as_default() as g:
        #由于所有的图结构都是由相同的模型定义的，因此不用重复加载相同的图结构
        #而是不用同一个模型恢复相同的图解钩，每次只是加载不同的变量值而已
        x = tf.placeholder(tf.float32, [None, mnist_forward.INPUT_NODE])
        y_ = tf.placeholder(tf.float32, [None, mnist_forward.OUT_NODE])
        y = mnist_forward.forward(x, None)

        ema=tf.train.ExponentialMovingAverage(mnist_backward.MOVING_AVERAGE_DECAY)
        ema_restore=ema.variables_to_restore()
        saver=tf.train.Saver(ema_restore)
        #定义加载变量，不修改加载原始参数，修改了加载滑动平均值，里面的参数不影响，因为滑动平均值早就计算好了

        correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
        accuracy=tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


        with tf.Session() as sess:
            ckpt=tf.train.get_checkpoint_state(mnist_backward.MODER_SAVE_PATH)
            print(ckpt)
            if ckpt and ckpt.all_model_checkpoint_paths:
                for path in ckpt.all_model_checkpoint_paths:
                    saver.restore(sess,path)
                    global_step=path.split('/')[-1].split('-')[-1]
                    accuracy_scare=sess.run(accuracy,feed_dict={x:mnist.test.images,y_:mnist.test.labels})
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




