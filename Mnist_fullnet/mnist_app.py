import tensorflow as tf
from  PIL import  Image
import numpy as np
import mnist_forward
import  mnist_backward
def restore_model(testpicArr):
    with tf.Graph().as_default() as g:
        x=tf.placeholder(tf.float32,[None,mnist_forward.INPUT_NODE])
        y=mnist_forward.forward(x,None)
        preValue=tf.argmax(y,1)

        variable_averages=tf.train.ExponentialMovingAverage(mnist_backward.MOVING_AVERAGE_DECAY)
        variable_to_restore=variable_averages.variables_to_restore()

        saver=tf.train.Saver(variable_to_restore)

        with tf.Session() as sess:
            ckpt=tf.train.get_checkpoint_state(mnist_backward.MODER_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess,ckpt.model_checkpoint_path)

                preValue=sess.run(preValue,feed_dict={x:testpicArr})
                return preValue
            else:
                print('no found model')
                return -1

def pre_pic(picName):
    img=Image.open(picName)
    relm=img.resize((28,28),Image.ANTIALIAS)
    im_arr=np.array(relm.convert('L'))
    threshold=50
    for i in range(28):
        for j in range(28):
            im_arr[i][j]=255-im_arr[i][j]
            if (im_arr[i][j]<threshold):
                im_arr[i][j]=0
            else:im_arr[i][j]=255
    nm_arr=im_arr.reshape(1,784)
    nm_arr=nm_arr.astype(np.float32)
    img_ready=np.multiply(nm_arr,1.0/255.0)
    return img_ready
def applicatoin():
    testNum=input('input the number of the picture:')
    for i in range(int(testNum)):
        testPic=input('the path of test picture:')
        testPicArr=pre_pic(testPic)
        preValue=restore_model(testPicArr)
        print('this number is',preValue)

def main():
    applicatoin()

if __name__=='__main__':
    main()