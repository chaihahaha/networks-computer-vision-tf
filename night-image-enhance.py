import numpy as np
import matplotlib.pyplot as plt
import scipy
from skimage.transform import resize
import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
import time

checkpoint_dir = './checkpoint/'
result_dir = './result/'

save_freq = 4
train_pics = 789
patches_num = 10
batch_size = 10
ckpt_freq = 1
learning_rate = 1e-6
lastepoch = 0

DEBUG = 0
if DEBUG == 1:
    save_freq = 2

VGG_MEAN = [103.939, 116.779, 123.68]

def initialize_uninitialized(sess):
    global_vars          = tf.global_variables()
    is_not_initialized   = sess.run([tf.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]

    print([str(i.name) for i in not_initialized_vars]) # only for testing
    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))

def vgg16(rgb, num_classes=1000,
           is_training=False,
           dropout_keep_prob=0.5,
           spatial_squeeze=False,
           scope='vgg_16',
           fc_conv_padding='VALID',
           global_pool=False):

    #start_time = time.time()
    print("build model started")
    rgb_scaled = rgb * 255.0

    # Convert RGB to BGR
    red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
#     assert red.get_shape().as_list()[1:] == [224, 224, 1]
#     assert green.get_shape().as_list()[1:] == [224, 224, 1]
#     assert blue.get_shape().as_list()[1:] == [224, 224, 1]
    bgr = tf.concat(axis=3, values=[
        blue - VGG_MEAN[0],
        green - VGG_MEAN[1],
        red - VGG_MEAN[2],
    ])
    #assert bgr.get_shape().as_list()[1:] == [224, 224, 3]

    with tf.variable_scope(scope, 'vgg_16', [bgr], reuse=tf.AUTO_REUSE) as sc:                     # 设定一个子网络的scope，便于之后指定需要训练的变量和导入权重
#         end_points={}
    # Collect outputs for conv2d, fully_connected and max_pool2d.
        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d]):
            net1 = slim.repeat(bgr, 2, slim.conv2d, 64, [3, 3], scope='conv1')
            net2 = slim.max_pool2d(net1, [2, 2], scope='pool1')
            net3 = slim.repeat(net2, 2, slim.conv2d, 128, [3, 3], scope='conv2')
            net4 = slim.max_pool2d(net3, [2, 2], scope='pool2')
            net5 = slim.repeat(net4, 3, slim.conv2d, 256, [3, 3], scope='conv3')
            net6 = slim.max_pool2d(net5, [2, 2], scope='pool3')
            net7 = slim.repeat(net6, 3, slim.conv2d, 512, [3, 3], scope='conv4')
            net8 = slim.max_pool2d(net7, [2, 2], scope='pool4')
            net9 = slim.repeat(net8, 3, slim.conv2d, 512, [3, 3], scope='conv5')
            
            
            
#             net = slim.max_pool2d(net, [2, 2], scope='pool5')

#           # Use conv2d instead of fully_connected layers.
#             net = slim.conv2d(net, 4096, [7, 7], padding=fc_conv_padding, scope='fc6')
#             net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
#                              scope='dropout6')
#             net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
#           # Convert end_points_collection into a end_point dict.
#             net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
#                                scope='dropout7')
#             net = slim.conv2d(net, num_classes, [1, 1],
#                               activation_fn=None,
#                               normalizer_fn=None,
#                               scope='fc8')
    return net2, net5, net9
    
def cov(x,y):
    mshape = x.shape
    #n,h,w,c
    x_bar = tf.reduce_mean(x, axis=[1,2,3])
    y_bar = tf.reduce_mean(y, axis=[1,2,3])
    x_bar = tf.einsum("i,jkl->ijkl",x_bar,tf.ones_like(x[0,:,:,:]))
    y_bar = tf.einsum("i,jkl->ijkl",y_bar,tf.ones_like(x[0,:,:,:]))
    return tf.reduce_mean((x-x_bar)*(y-y_bar), [1,2,3])
    

def sobel_loss(img1,img2):

    edge_1 = tf.math.abs(tf.image.sobel_edges(img1))
    edge_2 = tf.math.abs(tf.image.sobel_edges(img2))
    m_1 = tf.reduce_mean(edge_1)
    m_2 = tf.reduce_mean(edge_2)
    edge_bin_1 = tf.cast(edge_1>m_1, tf.float32)
    edge_bin_2 = tf.cast(edge_2>m_2, tf.float32)
    return tf.reduce_mean(tf.math.abs(edge_bin_1-edge_bin_2))

def generate_batch(patches_num,dark_img,gt_img):

    W = tf.shape(dark_img)[1]
    H = tf.shape(dark_img)[2]
    ps = 224
    

    #img_feed_in = tf.reshape(dark_img, [1, W, H, 3])
    #img_feed_gt = tf.reshape(gt_img, [1, 2 * W, 2 * H, 3])
    img_feed_in = dark_img
    img_feed_gt = gt_img
    

    input_patches = []
    gt_patches = []
    # random crop flip to generate patches
    for i in range(patches_num):
        xx = tf.random_uniform(dtype=tf.int32, minval=0, maxval=W - ps, shape=[1])
        yy = tf.random_uniform(dtype=tf.int32, minval=0, maxval=H - ps, shape=[1])

        input_patch = tf.slice(img_feed_in, [0, xx[0], yy[0], 0], [1, ps, ps, 3])

        #input_patch = img_feed_in[:, xx:xx + ps, yy:yy + ps, :]
        #gt_patch = img_feed_gt[:, xx * 2:xx * 2 + ps * 2, yy * 2:yy * 2 + ps * 2, :]
        gt_patch = tf.slice(img_feed_gt, [0, 2 * xx[0], 2 * yy[0], 0], [1, 2 * ps, 2 * ps, 3])


        input_patch, gt_patch = tf.cond(
            tf.less(tf.random_uniform([1], 0, 1)[0], 0.5), 
            true_fn=lambda: (tf.reverse(input_patch, [1]), tf.reverse(gt_patch, [1])), 
            false_fn=lambda: (input_patch, gt_patch))
        input_patch, gt_patch = tf.cond(
            tf.less(tf.random_uniform([1], 0, 1)[0], 0.5), 
            true_fn=lambda: (tf.reverse(input_patch, [2]), tf.reverse(gt_patch, [2])), 
            false_fn=lambda: (input_patch, gt_patch))
        input_patch, gt_patch = tf.cond(
            tf.less(tf.random_uniform([1], 0, 1)[0], 0.5), 
            true_fn=lambda: (tf.transpose(input_patch, (0, 2, 1, 3)), tf.transpose(gt_patch, (0, 2, 1, 3))), 
            false_fn=lambda: (input_patch, gt_patch))

        #input_patch = tf.minimum(input_patch, 1.0)
        input_patches.append(input_patch)
        gt_patches.append(gt_patch)
    
    return tf.concat(input_patches,0),tf.concat(gt_patches,0)
    




def lrelu(x):
    return tf.maximum(x * 0.2, x)


def upsample_and_concat(x1, x2, output_channels, in_channels):
    pool_size = 2
    deconv_filter = tf.Variable(tf.truncated_normal([pool_size, pool_size, output_channels, in_channels], stddev=0.02))
    deconv = tf.nn.conv2d_transpose(x1, deconv_filter, tf.shape(x2), strides=[1, pool_size, pool_size, 1])

    deconv_output = tf.concat([deconv, x2], 3)
    deconv_output.set_shape([None, None, None, output_channels * 2])

    return deconv_output


def network(input, scope="sid"):
    with tf.variable_scope(scope, "sid", reuse=tf.AUTO_REUSE) as sc:           # 设定一个子网络的scope，便于之后指定需要训练的变量和导入权重
        conv1 = slim.conv2d(input, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv1_1')
        conv1 = slim.conv2d(conv1, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv1_2')
        pool1 = slim.max_pool2d(conv1, [2, 2], padding='SAME')

        conv2 = slim.conv2d(pool1, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv2_1')
        conv2 = slim.conv2d(conv2, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv2_2')
        pool2 = slim.max_pool2d(conv2, [2, 2], padding='SAME')

        conv3 = slim.conv2d(pool2, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv3_1')
        conv3 = slim.conv2d(conv3, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv3_2')
        pool3 = slim.max_pool2d(conv3, [2, 2], padding='SAME')

        conv4 = slim.conv2d(pool3, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv4_1')
        conv4 = slim.conv2d(conv4, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv4_2')
        pool4 = slim.max_pool2d(conv4, [2, 2], padding='SAME')

        conv5 = slim.conv2d(pool4, 512, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv5_1')
        conv5 = slim.conv2d(conv5, 512, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv5_2')

        up6 = upsample_and_concat(conv5, conv4, 256, 512)
        conv6 = slim.conv2d(up6, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv6_1')
        conv6 = slim.conv2d(conv6, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv6_2')

        up7 = upsample_and_concat(conv6, conv3, 128, 256)
        conv7 = slim.conv2d(up7, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv7_1')
        conv7 = slim.conv2d(conv7, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv7_2')

        up8 = upsample_and_concat(conv7, conv2, 64, 128)
        conv8 = slim.conv2d(up8, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv8_1')
        conv8 = slim.conv2d(conv8, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv8_2')

        up9 = upsample_and_concat(conv8, conv1, 32, 64)
        conv9 = slim.conv2d(up9, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv9_1')
        conv9 = slim.conv2d(conv9, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv9_2')

        conv10 = slim.conv2d(conv9, 12, [1, 1], rate=1, activation_fn=None, scope='g_conv10')
        out = tf.depth_to_space(conv10, 2)

#     ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
#     if ckpt:
#         print('loaded ' + ckpt.model_checkpoint_path)
#         tf.contrib.framework.init_from_checkpoint(ckpt.model_checkpoint_path, {"sid/":"sid/"})
    return out




sess = tf.Session()

in_image = tf.placeholder(tf.float32, [None, None, None, 3])
gt_image = tf.placeholder(tf.float32, [None, None, None, 3])
input_patches, gt_patches = generate_batch(patches_num, in_image, gt_image)
output_patches = network(input_patches)


out_image = network(in_image[0:1,:,:,:])[0,:,:,:]
vgg_o = vgg16(output_patches)
vgg_g = vgg16(gt_patches)
loss0 = tf.reduce_mean(tf.abs(vgg_o[0] - vgg_g[0]))
loss1 = tf.reduce_mean(tf.abs(vgg_o[1] - vgg_g[1]))
loss2 = tf.reduce_mean(tf.abs(vgg_o[2] - vgg_g[2]))
G_loss = loss0 #+ loss1 + loss2 #+ tf.reduce_mean(tf.abs(output_patches - gt_patches))

weight = tf.reduce_mean([v for v in tf.trainable_variables() if v.name == "sid/g_conv1_1/weights:0"])




t_vars = tf.trainable_variables()
lr = tf.placeholder(tf.float32)

train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "sid")  # 设定需要训练的变量，通过scope指定
# print(train_vars)
G_opt = tf.train.AdamOptimizer(learning_rate=lr).minimize(G_loss, var_list=train_vars)   # var_list指定需要优化的变量

saver_sid = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "sid"))
saver_vgg = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "vgg_16"))
sess.run(tf.global_variables_initializer())                                      # 全局变量初始化
saver_vgg.restore(sess, "vgg_16.ckpt")

ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
if ckpt:
    print('loaded ' + ckpt.model_checkpoint_path)
    saver_sid.restore(sess, ckpt.model_checkpoint_path) 

    
g_loss = np.zeros((5000, 1))

if not os.path.isdir(result_dir):
    os.makedirs(result_dir)


    
for epoch in range(lastepoch, 4001):
    cnt = 0
    if epoch > 2000:
        learning_rate = 1e-7
    batches_num = train_pics//batch_size
    for batch in range(batches_num):#np.random.permutation(train_pics):
        st = time.time()
        cnt += 1
        
#         choice = np.random.randint(1, train_pics + 1)
        dark_img = []
        gt_img = []
        for ind in np.random.permutation(batch_size):
            dark_img.append(plt.imread("Low/low"+"{0:05}".format((batch*batch_size + ind)%train_pics + 1)+".png")[np.newaxis, :,:,:])
            gt_tmp = plt.imread("Normal/normal"+"{0:05}".format((batch*batch_size + ind)%train_pics + 1)+".png")
            gt_img.append(resize(gt_tmp, (gt_tmp.shape[0]*2,gt_tmp.shape[1]*2))[np.newaxis, :,:,:])
        dark_img = np.concatenate(dark_img, 0)
        gt_img = np.concatenate(gt_img, 0)

        _, G_current, output , weight_f, los0, los1, los2= sess.run([G_opt, G_loss, out_image, weight, loss0, loss1, loss2],
                                feed_dict={in_image: dark_img, gt_image:gt_img, lr: learning_rate})

        output = np.minimum(np.maximum(output, 0), 1)
        g_loss[batch] = G_current

        print("%d %d Loss=%.5f Time=%.5f Weight=%.5f Loss0=%.5f Loss1=%.5f Loss2=%.5f" % (epoch, cnt, np.mean(g_loss[np.where(g_loss)]), time.time() - st, weight_f*1e3, los0, los1, los2))

    if epoch % save_freq == 0:
        print("saving result jpg")
        if not os.path.isdir(result_dir + '%04d' % epoch):
            os.makedirs(result_dir + '%04d' % epoch)
        input_pth_tmp = input_patches[0,:,:,:]
        temp = np.concatenate((gt_img[0,:,:,:], output, resize(dark_img[0,:,:,:], (dark_img.shape[1]*2,dark_img.shape[2]*2))), axis=1)
        scipy.misc.toimage(temp * 255, high=255, low=0, cmin=0, cmax=255).save(
            result_dir + '%04d/%05d_00_train.jpg' % (epoch, batch))
            
    if epoch % ckpt_freq == 0:
        saver_sid.save(sess, checkpoint_dir + 'model_'+str(np.mean(g_loss[np.where(g_loss)]))+'.ckpt')


