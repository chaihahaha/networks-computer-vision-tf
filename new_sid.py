import numpy as np
import matplotlib.pyplot as plt
import scipy
from skimage.transform import resize
import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
import time
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.plugin.tf as dali_tf

checkpoint_dir = './checkpoint/'
result_dir = './result/'
dark_img_dir = './images_dark/'
gt_img_dir = './images_gt/'
use_model = 'sid'
print_weight = 'g_conv1_1/weight:0'

save_freq = 10
train_pics = 789
patches_num = 2
batch_size = 16  # maximum 32
ckpt_freq = 10
learning_rate = 1e-4
lastepoch = 0
model_size = 100

DEBUG = 0
if DEBUG == 1:
    save_freq = 2

VGG_MEAN = [103.939, 116.779, 123.68]

class SimplePipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, img_dir):
        super(SimplePipeline, self).__init__(batch_size, num_threads, device_id, seed = 12)
        self.input = ops.FileReader(file_root = img_dir)
        # instead of path to file directory file with pairs image_name image_label_value can be provided
        # self.input = ops.FileReader(file_root = image_dir, file_list = image_dir + '/file_list.txt')
        self.decode = ops.ImageDecoder(device = 'mixed', output_type = types.RGB)
    def define_graph(self):
        pngs, _ = self.input()
        images = self.decode(pngs)
        return images
    
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
#     print("build model started")
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
            net10 = slim.max_pool2d(net9, [2, 2], scope='pool5')

#           # Use conv2d instead of fully_connected layers.
            net11 = slim.conv2d(net10, 4096, [7, 7], padding=fc_conv_padding, scope='fc6')
#             net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
#                              scope='dropout6')
            net12 = slim.conv2d(net11, 4096, [1, 1], scope='fc7')
#           # Convert end_points_collection into a end_point dict.
#             net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
#                                scope='dropout7')
#             net = slim.conv2d(net, num_classes, [1, 1],
#                               activation_fn=None,
#                               normalizer_fn=None,
#                               scope='fc8')
    return net12
    
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
    B = tf.shape(dark_img)[0]
    H = tf.shape(dark_img)[1]
    W = tf.shape(dark_img)[2]
    ps = 300
    
    img_feed_gt = gt_img

    img_feed_in = dark_img
#     print(img_feed_gt)
#     print(img_feed_in)

    input_patches = []
    gt_patches = []
    # random crop flip to generate patches
    for i in range(patches_num):
        xx = tf.random_uniform(dtype=tf.int32, minval=0, maxval=H - ps, shape=[1])
        yy = tf.random_uniform(dtype=tf.int32, minval=0, maxval=W - ps, shape=[1])

        input_patch = tf.slice(img_feed_in, [0, xx[0], yy[0], 0], [B, ps, ps, 3])

        #input_patch = img_feed_in[:, xx:xx + ps, yy:yy + ps, :]
        #gt_patch = img_feed_gt[:, xx * 2:xx * 2 + ps * 2, yy * 2:yy * 2 + ps * 2, :]
        gt_patch = tf.slice(img_feed_gt, [0,  xx[0],  yy[0], 0], [B,  ps,  ps, 3])


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
    input_patches = tf.concat(input_patches,0)
    gt_patches = tf.concat(gt_patches,0)
    
    return input_patches, gt_patches
    




def lrelu(x):
    return tf.maximum(x * 0.2, x)


def upsample_and_concat(x1, x2, output_channels, in_channels):
    #h_x2==2 * h_x1  x1:in_channels x2:out_channels
    pool_size = 2
    deconv_filter = tf.Variable(tf.truncated_normal([pool_size, pool_size, output_channels, in_channels], stddev=0.02))
    deconv = tf.nn.conv2d_transpose(x1, deconv_filter, tf.shape(x2), strides=[1, pool_size, pool_size, 1])

    deconv_output = tf.concat([deconv, x2], 3)
    deconv_output.set_shape([None, None, None, output_channels * 2])

    return deconv_output

    
def my_conv2d(x,idepth, odepth,size):
    conv1_weights = tf.Variable(tf.random_normal([size,size,idepth,odepth]))
    conv1_biases = tf.Variable(tf.random_normal([odepth]))
    conv = tf.nn.conv2d(x,conv1_weights,strides=[1, 1, 1, 1],padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))
    return relu



def network(input):
    H = tf.shape(input)[1]
    W = tf.shape(input)[2]
    with tf.variable_scope("sid", "sid", reuse=tf.AUTO_REUSE) as sc:           # 设定一个子网络的scope，便于之后指定需要训练的变量和导入权重
        
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

        conv10 = slim.conv2d(conv9, 3, [1, 1], rate=1, activation_fn=None, scope='g_conv10')
#         out = tf.depth_to_space(conv10, 2)
    return conv10
def cosine_distance(t1, t2):
    t1_len = tf.sqrt(tf.reduce_mean(t1*t1, [1,2,3]))
    t2_len = tf.sqrt(tf.reduce_mean(t2*t2, [1,2,3]))
    products = tf.reduce_mean(t1*t2, [1,2,3])
    cosine_dis = 1-products/(t1_len*t2_len)
    return cosine_dis
def variance_distance(t1, t2):
    t1_len = tf.sqrt(tf.reduce_mean(t1*t1, [1,2,3]))
    t2_len = tf.sqrt(tf.reduce_mean(t2*t2, [1,2,3]))
    t1_mean = tf.reduce_mean(t1, [1,2,3])
    t2_mean = tf.reduce_mean(t2, [1,2,3])
    t1_var = t1_len**2 - t1_mean**2
    t2_var = t2_len**2 - t2_mean**2
    variance_dis = tf.abs(t1_var - t2_var)
    return variance_dis
def distance(t1, t2):
    k = 1.0
    delta = 3.0
    return  tf.reduce_mean(tf.exp(k * tf.abs(t1 - t2) + delta) - tf.exp(delta) , [1,2,3])
    


sess = tf.Session()

dark_pipe = SimplePipeline(batch_size, 1, 0, dark_img_dir)
gt_pipe = SimplePipeline(batch_size, 1, 0, gt_img_dir)
daliop = dali_tf.DALIIterator()
in_image= daliop(pipeline = dark_pipe, shapes=[[batch_size,400,600,3]],dtypes=[tf.uint8])
in_image=tf.to_float(in_image[0])/255.0

gt_image= daliop(pipeline = gt_pipe, shapes=[[batch_size,400,600,3]],dtypes=[tf.uint8])
gt_image=tf.to_float(gt_image[0])/255.0

print(in_image,gt_image)

input_patches, gt_patches = generate_batch(patches_num, in_image, gt_image)
output_patches = network(input_patches)
out_max = tf.reduce_max(output_patches)
out_min = tf.reduce_min(output_patches)
out_image = network(in_image[0:1,:,:,:])[0,:,:,:]
debug_in = tf.reduce_mean(in_image[0:1,:,:,:])

o_debug = tf.reduce_mean(output_patches)
o_img_debug = tf.reduce_mean(out_image)
i_debug = tf.reduce_mean(input_patches)
g_debug = tf.reduce_mean(gt_patches)
loss_dis = tf.abs(output_patches- gt_patches)
G_loss = tf.reduce_mean(loss_dis) #+ loss1 + loss2 #+ tf.reduce_mean(tf.abs(output_patches - gt_patches))
print("I will print the following variable weight for you")
print([v for v in tf.trainable_variables() if v.name == use_model + '/' + print_weight])
weight = tf.reduce_mean([v for v in tf.trainable_variables() if v.name == use_model + '/' + print_weight])




t_vars = tf.trainable_variables()
lr = tf.placeholder(tf.float32)

train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, use_model)  # 设定需要训练的变量，通过scope指定
# print(train_vars)
G_opt = tf.train.AdamOptimizer(learning_rate=lr).minimize(G_loss, var_list=train_vars)   # var_list指定需要优化的变量

saver_sid = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, use_model))
sess.run(tf.global_variables_initializer())                                      # 全局变量初始化


ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
if ckpt:
    print('loaded ' + ckpt.model_checkpoint_path)
    saver_sid.restore(sess, ckpt.model_checkpoint_path) 

    
g_loss = np.zeros((5000, 1))

if not os.path.isdir(result_dir):
    os.makedirs(result_dir)



for epoch in range(lastepoch, 4001):
    cnt = 0
    if epoch > 200:
        learning_rate = 1e-5
    batches_num = train_pics//batch_size
    for batch in range(batches_num):#np.random.permutation(train_pics):
        st = time.time()
        cnt += 1
        
#         choice = np.random.randint(1, train_pics + 1)

        
#         print("np dark gt shape")
#         print(dark_img.shape)
#         print(gt_img.shape)
        
        _, G_current, output , weight_f, o_d , i_d, g_d, o_ma, o_mi, o_img, d_i, gt_img, dark_img= sess.run([G_opt, G_loss, out_image, weight, o_debug, i_debug, g_debug, out_max, out_min, o_img_debug, debug_in, gt_image, in_image],
                                feed_dict={ lr: learning_rate})

        output = np.minimum(np.maximum(output, 0), 1)
        g_loss[batch] = G_current

        print("%d %d Loss=%.5f Time=%.5f Weight=%.5f LossCurrent=%.5f OutputMean=%.5f InputMean=%.5f GTMean=%.5f OMax=%.2f OMin=%.2f OIMG=%.2f IIMG=%.2f" % (epoch, cnt, np.mean(g_loss[np.where(g_loss)]), time.time() - st, weight_f*1e3, G_current, o_d, i_d, g_d, o_ma, o_mi, o_img, d_i))

    if epoch % save_freq == 0:
        print("saving result " + result_dir + '%04d/%05d_00_train.jpg' % (epoch, batch))
        if not os.path.isdir(result_dir + '%04d' % epoch):
            os.makedirs(result_dir + '%04d' % epoch)
       
        
        temp = np.concatenate((gt_img[0,:,:,:], output, dark_img[0,:,:,:]), axis=1)
        scipy.misc.toimage(temp * 255, high=255, low=0, cmin=0, cmax=255).save(
            result_dir + '%04d/%05d_00_train.jpg' % (epoch, batch))
            
    if epoch % ckpt_freq == 0:
        print("saving checkpoint...")
        saver_sid.save(sess, checkpoint_dir + 'model_'+str(np.mean(g_loss[np.where(g_loss)]))+'.ckpt')


