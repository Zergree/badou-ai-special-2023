import tensorflow as tf
import numpy as np
import time
import math
import Cifar10_data

max_steps=4000
batch_size=100
num_examples_for_eval=10000
data_dir="Cifar_data/cifar-10-batches-bin"

def variable_with_weight_loss(shape,stddev,w1):
    var=tf.Variable(tf.truncated_normal(shape,stddev=stddev))
    if w1 is not None:
        weights_loss=tf.multiply(tf.nn.l2_loss(var),w1,name='weights_loss')
        tf.add_to_collection('losses',weights_loss)
    return var

'''
w1 参数是用来控制权重衰减的强度的，它代表了正则化项的权重。正则化项的作用是用来约束模型的参数，防止过拟合。当 w1 为 0 时，表示不对该变量应用权重衰减，即不添加正则化项。而当 w1 不为 0 时，表示对该变量应用权重衰减，并且 w1 的值越大，权重衰减的强度就越大，模型的参数会更加受到约束。

在实际应用中，可以根据模型的复杂度、数据集的大小以及训练集和验证集的表现情况来调节 w1 的取值。如果模型出现了过拟合的现象，可以尝试增大 w1 的值，以加强正则化的作用，从而减少过拟合。相反，如果模型出现了欠拟合的现象，可以考虑减小 w1 的值，或者将其设为 0，以允许模型更自由地学习数据的特征。

当 w1 为 0 时，权重衰减的正则化项不参与损失函数的计算，因此在反向传播过程中，不会对该变量的梯度进行修改。这意味着该变量的更新不受权重衰减的影响，但仍然会受到损失函数的梯度影响而进行更新
'''

images_train,labels_train=Cifar10_data.inputs(data_dir=data_dir,batch_size=batch_size,distorted=True)
images_test,labels_test=Cifar10_data.inputs(data_dir=data_dir,batch_size=batch_size,distorted=None)


x=tf.placeholder(tf.float32,[batch_size,24,24,3])
y_=tf.placeholder(tf.int32,[batch_size])




# 第一层卷积
kernel1=variable_with_weight_loss(shape=[5,5,3,64],stddev=5e-2,w1=0.0)
conv1=tf.nn.conv2d(x,kernel1,[1,1,1,1],padding='SAME')
bias1=tf.Variable(tf.constant(0.0,shape=[64]))
relu1=tf.nn.relu(tf.nn.bias_add(conv1,bias1))
pool1=tf.nn.max_pool(relu1,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME')

# 第二层卷积
kernel2=variable_with_weight_loss(shape=[5,5,64,64],stddev=5e-2,w1=0.0)
conv2=tf.nn.conv2d(pool1,kernel2,[1,1,1,1],padding='SAME')
bias2=tf.Variable(tf.constant(0.1,shape=[64]))
relu2=tf.nn.relu(tf.nn.bias_add(conv2,bias2))
pool2=tf.nn.max_pool(relu2,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME')

# 在全连接之前进行拍扁
reshape=tf.reshape(pool2,[batch_size,-1])
dim=reshape.get_shape()[1].value

#建立第一个全连接
weight1=variable_with_weight_loss(shape=[dim,384],stddev=0.04,w1=0.004)
fc_bias1=tf.Variable(tf.constant(0.1,shape=[384]))
fc_1=tf.nn.relu(tf.matmul(reshape,weight1)+fc_bias1)

#建立第二个全连接
weight2=variable_with_weight_loss(shape=[384,192],stddev=0.04,w1=0.004)
fc_bias2=tf.Variable(tf.constant(0.1,shape=[192]))
local4=tf.nn.relu(tf.matmul(fc_1,weight2)+fc_bias2)

#建立第三个全连接
weight3=variable_with_weight_loss(shape=[192,10],stddev=1/192.0,w1=0.0)
fc_bias3=tf.Variable(tf.constant(0.1,shape=[10]))
result=tf.add(tf.matmul(local4,weight3),fc_bias3)

#计算损失，，包括权重参数的正则化损失和交叉熵损失
cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=result,labels=tf.cast(y_,tf.int64))

weights_with_l2_loss=tf.add_n(tf.get_collection("losses"))
loss=tf.reduce_mean(cross_entropy)+weights_with_l2_loss

train_op=tf.train.AdamOptimizer(1e-3).minimize(loss)

top_k_op=tf.nn.in_top_k(result,y_,1)

'''
tf.nn.in_top_k() 函数的作用是判断模型预测结果中的前 k 个最大值是否包含真实标签。如果真实标签在前 k 个最大值中，则认为预测是正确的；否则认为预测是错误的。这个函数常用于分类任务中，特别是多类别分类任务中的评估过程
'''

init_op=tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)

    tf.train.start_queue_runners()

#每隔100step会计算并展示当前的loss、每秒钟能训练的样本数量、以及训练一个batch数据所花费的时间
    for step in range (max_steps):
        start_time=time.time()
        image_batch,label_batch=sess.run([images_train,labels_train])
        _,loss_value=sess.run([train_op,loss],feed_dict={x:image_batch,y_:label_batch})
        duration=time.time() - start_time

        if step % 100 == 0:
            examples_per_sec=batch_size / duration
            sec_per_batch=float(duration)
            print("step %d,loss=%.2f(%.1f examples/sec;%.3f sec/batch)"%(step,loss_value,examples_per_sec,sec_per_batch))

#计算最终的正确率
    num_batch=int(math.ceil(num_examples_for_eval/batch_size))  #math.ceil()函数用于求整
    true_count=0
    total_sample_count=num_batch * batch_size

    #在一个for循环里面统计所有预测正确的样例个数
    for j in range(num_batch):
        image_batch,label_batch=sess.run([images_test,labels_test])
        predictions=sess.run([top_k_op],feed_dict={x:image_batch,y_:label_batch})
        true_count += np.sum(predictions)

    #打印正确率信息
    print("accuracy = %.3f%%"%((true_count/total_sample_count) * 100))












