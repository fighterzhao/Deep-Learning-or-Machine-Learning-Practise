import tensorflow as tf
import mnist_forward
import os
from tensorflow.examples.tutorials.mnist import input_data

BATCH_SIZE=200
LEARNING_RATE_BASE=0.1
LEARNING_RATE_DECAY=0.99
REGULARIZER=0.001
STEPS=20000
#只代表这个模型每一次训练最多训练这么多轮，可以重复进行，最后总轮数为分批次的轮数和
MOVING_AVERAGE_DECAY=0.99
MODER_SAVE_PATH='./model/'
MODER_NAME='mnist_model'

def backward(mnist):
    x=tf.placeholder(tf.float32,[None,mnist_forward.INPUT_NODE])
    y_=tf.placeholder(tf.float32,[None,mnist_forward.OUT_NODE])
    y=mnist_forward.forward(x,REGULARIZER)
    global_step=tf.Variable(0,trainable=False)

    ce=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))
    cem=tf.reduce_mean(ce)
    loss=cem+tf.add_n(tf.get_collection('losses'))

    learning_rate=tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples/BATCH_SIZE,
        LEARNING_RATE_DECAY,
        staircase=True
    )

    train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)

    ema=tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
    ema_op=ema.apply(tf.trainable_variables())
    #衰减速率（min(decay,1+step/10+step），可以初期使用较小的，后期使用较大一点的，目的是维持一个影子变量，这个影子变量本次的取值与
    #上一次的影子变量值与本次的参数值有关，目的是初期让影子变化快一点（参数不太准确），后期倾向于稳定变化，随着迭代一直更新不影响训练过程
    #参与趋于稳定时候，影子影响大一点，更新慢一点，参数前期不稳定，应该更新快一点，所以decay应该随着步数变大
    #影子更新不影响参数，但是参数更新会影响影子
    #华东平均可以每走一步更新一次影子，也可以做很多步，更新一次影子

    #开始影子更新快一点，后期影子更新慢一点，每一步都对影子进行更新，也可以隔好多步对影子进行更新
    with tf.control_dependencies([train_step,ema_op]):
        train_op=tf.no_op(name='train')

    saver=tf.train.Saver()

    with tf.Session() as sess:
        init_op=tf.global_variables_initializer()
        sess.run(init_op)

        ckpt=tf.train.get_checkpoint_state(MODER_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess,ckpt.model_checkpoint_path)
        for i in range(STEPS):
            xs,ys=mnist.train.next_batch(BATCH_SIZE)
            _,loss_value,step=sess.run([train_op,loss,global_step],feed_dict={x:xs,y_:ys})
            if i % 2000==0:
                print(i,step,loss_value)
                # saver.save(sess,os.path.join(MODER_SAVE_PATH,MODER_NAME),global_step=global_step)


def main():
    mnist=input_data.read_data_sets('./data/',one_hot=True)
    backward(mnist)

if __name__ =='__main__':
    main()











