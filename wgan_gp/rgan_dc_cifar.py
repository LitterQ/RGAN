import os, sys
sys.path.append(os.getcwd())


import time

import numpy as np
import scipy
import scipy.io
import tensorflow as tf

import tflib as lib
import tflib.ops.linear
import tflib.ops.conv2d
import tflib.ops.batchnorm
import tflib.ops.deconv2d
import tflib.save_images
import tflib.cifar10
import tflib.inception_score
import tflib.plot

EPSILON = 0.01
EPSILON_dis = 1
checkpoint = './result/cifar_dcgan_fea_v6_5'
# Download CIFAR-10 (Python version) at
# https://www.cs.toronto.edu/~kriz/cifar.html and fill in the path to the
# extracted files here!
DATA_DIR = './cifar-10-python'
if len(DATA_DIR) == 0:
    raise Exception('Please specify path to data directory in gan_cifar.py!')

MODE = 'dcgan' # Valid options are dcgan, wgan, or wgan-gp
DIM = 128 # This overfits substantially; you're probably better off with 64
LAMBDA = 10 # Gradient penalty lambda hyperparameter
CRITIC_ITERS = 2 # How many critic iterations per generator iteration
BATCH_SIZE = 64 # Batch size
ITERS = 200000 # How many generator iterations to train for
OUTPUT_DIM = 3072 # Number of pixels in CIFAR10 (3*32*32)

lib.print_model_settings(locals().copy())

def get_normalized_vector(d):
    d /= (1e-12 + tf.reduce_max(tf.abs(d), range(1, len(d.get_shape())), keep_dims=True))
    d /= tf.sqrt(1e-6 + tf.reduce_sum(tf.pow(d, 2.0), range(1, len(d.get_shape())), keep_dims=True))
    return d

def generate_adversarial_perturbation(x):
    logit_g, aa, _ = Generator(BATCH_SIZE,x)
    logit_d = Discriminator(logit_g)[0]
    #loss = -tf.reduce_mean(logit_d)
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logit_d, labels=tf.ones_like(logit_d)))
    grad = tf.gradients(loss, [x], aggregation_method=2)[0]
    grad = tf.stop_gradient(grad)
    return EPSILON * get_normalized_vector(grad)

def adversarial_loss(x):
    r_adv = generate_adversarial_perturbation(x)
    logit_g,bb,fea_bb = Generator(BATCH_SIZE,x + r_adv)
    logit_d = Discriminator(logit_g)[0]
    #loss = -tf.reduce_mean(logit_d)
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logit_d, labels=tf.ones_like(logit_d)))

    return loss,r_adv,fea_bb

def generate_adversarial_perturbation_dis(x, real):
    #logit_g, aa, _ = Generator(BATCH_SIZE,x)
    logit_d = Discriminator(x)[0]
    logit_real = Discriminator(real)[0]
    #loss = tf.reduce_mean(logit_d) - tf.reduce_mean(logit_real)
    loss =  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logit_d, labels=tf.zeros_like(logit_d)))
    loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logit_real, labels=tf.ones_like(logit_real)))
    loss /= 2.
    grad = tf.gradients(loss, [x], aggregation_method=2)[0]
    #grad_real = tf.gradients(loss, [real], aggregation_method=2)[0]
    grad = tf.stop_gradient(grad)
    #grad_real = tf.stop_gradient(grad_real)
    return EPSILON_dis * get_normalized_vector(grad)#, EPSILON_dis * get_normalized_vector(grad_real)

def adversarial_loss_dis(x,real):
    r_adv = generate_adversarial_perturbation_dis(x, real)
    #logit_g,bb,fea_bb = Generator(BATCH_SIZE,x + r_adv)
    logit_d_org  = Discriminator(x)[0]
    #logit_d_org = tf.stop_gradient(logit_d_org)
    logit_d = Discriminator(x+r_adv)[0]
    logit_real = Discriminator(real)[0]
    loss = tf.reduce_sum(tf.square(logit_d-logit_d_org))
    #loss = tf.reduce_mean(loss**2)
    #loss = tf.sqrt(tf.reduce_mean(tf.square(logit_d-logit_d_org)))
    return loss,r_adv#,real_adv

def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha*x, x)

def ReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(name+'.Linear', n_in, n_out, inputs)
    return tf.nn.relu(output)

def LeakyReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(name+'.Linear', n_in, n_out, inputs)
    return LeakyReLU(output)

def Generator(n_samples, noise=None):
    if noise is None:
        noise = tf.random_normal([n_samples, 128])

    output = lib.ops.linear.Linear('Generator.Input', 128, 4*4*4*DIM, noise)
    output = lib.ops.batchnorm.Batchnorm('Generator.BN1', [0], output)
    output = tf.nn.relu(output)
    output = tf.reshape(output, [-1, 4*DIM, 4, 4])

    output = lib.ops.deconv2d.Deconv2D('Generator.2', 4*DIM, 2*DIM, 5, output)
    output = lib.ops.batchnorm.Batchnorm('Generator.BN2', [0,2,3], output)
    output = tf.nn.relu(output)
    fea = output

    output = lib.ops.deconv2d.Deconv2D('Generator.3', 2*DIM, DIM, 5, output)
    output = lib.ops.batchnorm.Batchnorm('Generator.BN3', [0,2,3], output)
    output = tf.nn.relu(output)

    output = lib.ops.deconv2d.Deconv2D('Generator.5', DIM, 3, 5, output)

    output = tf.tanh(output)

    return (tf.reshape(output, [-1, OUTPUT_DIM]), noise, fea)

def Discriminator(inputs):
    output = tf.reshape(inputs, [-1, 3, 32, 32])

    output = lib.ops.conv2d.Conv2D('Discriminator.1', 3, DIM, 5, output, stride=2)
    output = LeakyReLU(output)

    output = lib.ops.conv2d.Conv2D('Discriminator.2', DIM, 2*DIM, 5, output, stride=2)
    if MODE != 'wgan-gp':
        output = lib.ops.batchnorm.Batchnorm('Discriminator.BN2', [0,2,3], output)
    output = LeakyReLU(output)

    output = lib.ops.conv2d.Conv2D('Discriminator.3', 2*DIM, 4*DIM, 5, output, stride=2)
    if MODE != 'wgan-gp':
        output = lib.ops.batchnorm.Batchnorm('Discriminator.BN3', [0,2,3], output)
    output = LeakyReLU(output)
    dis_fea = output

    output = tf.reshape(output, [-1, 4*4*4*DIM])
    output = lib.ops.linear.Linear('Discriminator.Output', 4*4*4*DIM, 1, output)

    return tf.reshape(output, [-1]), dis_fea

real_data_int = tf.placeholder(tf.int32, shape=[BATCH_SIZE, OUTPUT_DIM])
real_data = 2*((tf.cast(real_data_int, tf.float32)/255.)-.5)
fake_data,noise,fea_org = Generator(BATCH_SIZE)

disc_real,_ = Discriminator(real_data)
disc_fake, fake_fea = Discriminator(fake_data)

gen_params = lib.params_with_name('Generator')
disc_params = lib.params_with_name('Discriminator')

if MODE == 'wgan':
    gen_cost = -tf.reduce_mean(disc_fake)
    disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

    gen_train_op = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(gen_cost, var_list=gen_params)
    disc_train_op = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(disc_cost, var_list=disc_params)

    clip_ops = []
    for var in disc_params:
        clip_bounds = [-.01, .01]
        clip_ops.append(
            tf.assign(
                var,
                tf.clip_by_value(var, clip_bounds[0], clip_bounds[1])
            )
        )
    clip_disc_weights = tf.group(*clip_ops)

elif MODE == 'wgan-gp':
    # Standard WGAN loss
    gen_cost = -tf.reduce_mean(disc_fake)
    disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

    # Gradient penalty
    alpha = tf.random_uniform(
        shape=[BATCH_SIZE,1],
        minval=0.,
        maxval=1.
    )
    differences = fake_data - real_data
    interpolates = real_data + (alpha*differences)
    gradients = tf.gradients(Discriminator(interpolates)[0], [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
    gradient_penalty = tf.reduce_mean((slopes-1.)**2)
    disc_cost += LAMBDA*gradient_penalty


    ad_loss,r_adv,fea_ad = adversarial_loss(noise)
    result_noise,_,_ = Generator(BATCH_SIZE,noise+r_adv)

    dis_noise_fake, dis_noise_fea = Discriminator(result_noise)
    fake_fea1 = tf.stop_gradient(fake_fea)
    dis_mse = tf.reduce_mean(tf.square(dis_noise_fea-fake_fea1))

    fea_org1 = tf.stop_gradient(fea_org)
    fea_mse = tf.reduce_mean(tf.square(fea_org-fea_ad))

    differences_dis = result_noise - real_data
    interpolates_dis = real_data + (alpha * differences_dis)
    gradients_dis = tf.gradients(Discriminator(interpolates_dis)[0], [interpolates_dis])[0]
    slopes_dis = tf.sqrt(tf.reduce_sum(tf.square(gradients_dis), reduction_indices=[1]))
    gradient_penalty_dis = tf.reduce_mean((slopes_dis - 1.) ** 2)

    disc_cost_noise = tf.reduce_mean(dis_noise_fake) - tf.reduce_mean(disc_real)
    disc_cost_noise = disc_cost_noise + gradient_penalty_dis
    #disc_cost = disc_cost_ora + (ad_loss**2)
    disc_cost_ad = 1 * disc_cost + 1 * disc_cost_noise
    #disc_cost =  disc_cost_noise
    gen_cost_ad = 1 * gen_cost + 1 * ad_loss + 10 * dis_mse

    gen_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(gen_cost_ad, var_list=gen_params)
    disc_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(disc_cost_ad, var_list=disc_params)

elif MODE == 'dcgan':
    gen_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake, labels=tf.ones_like(disc_fake)))
    disc_cost =  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake, labels=tf.zeros_like(disc_fake)))
    disc_cost += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_real, labels=tf.ones_like(disc_real)))
    disc_cost /= 2.

    alpha = tf.random_uniform(
        shape=[BATCH_SIZE,1],
        minval=0.,
        maxval=1.
    )

    ad_loss,r_adv,fea_ad = adversarial_loss(noise)
    result_noise,_,_ = Generator(BATCH_SIZE,noise+r_adv)

    dis_noise_fake, dis_noise_fea = Discriminator(result_noise)
    fake_fea1 = tf.stop_gradient(fake_fea)
    dis_mse = tf.reduce_mean(tf.square(dis_noise_fea-fake_fea1))

    fea_org1 = tf.stop_gradient(fea_org)
    fea_mse = tf.reduce_mean(tf.square(fea_org-fea_ad))


    #disc_cost_noise = tf.reduce_mean(dis_noise_fake) - tf.reduce_mean(disc_real)
    disc_cost_noise = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_noise_fake, labels=tf.zeros_like(dis_noise_fake)))

    differences = fake_data - real_data
    interpolates = real_data + (alpha * differences)
    differences_dis = result_noise - real_data
    interpolates_dis = real_data + (alpha * differences_dis)
    ad_loss_dis, _ = adversarial_loss_dis(interpolates,real_data)
    ad_loss_dis_noise, _ = adversarial_loss_dis(interpolates_dis, real_data)
    #disc_cost_noise = disc_cost_noise + gradient_penalty_dis
    #disc_cost = disc_cost_ora + (ad_loss**2)
    disc_cost_ad = 1 * disc_cost + 0.3 * disc_cost_noise + 0.1 * ad_loss_dis + 0.1 * ad_loss_dis_noise
    #disc_cost =  disc_cost_noise
    gen_cost_ad = 1 * gen_cost + 1 * ad_loss + 0 * dis_mse

    gen_train_op = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(gen_cost_ad,
                                                                                  var_list=lib.params_with_name('Generator'))
    disc_train_op = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(disc_cost_ad,
                                                                                   var_list=lib.params_with_name('Discriminator.'))

# For generating samples
fixed_noise_128 = tf.constant(np.random.normal(size=(128, 128)).astype('float32'))
fixed_noise_samples_128,_,_ = Generator(128, noise=fixed_noise_128)
def generate_image(frame, true_dist):
    samples = session.run(fixed_noise_samples_128)
    samples = ((samples+1.)*(255./2)).astype('int32')
    lib.save_images.save_images(samples.reshape((128, 3, 32, 32)), checkpoint + '/samples/samples_{}.jpg'.format(frame))

# For calculating inception score
samples_100,_,_ = Generator(100)
def get_inception_score():
    all_samples = []
    for i in xrange(10):
        all_samples.append(session.run(samples_100))
    all_samples = np.concatenate(all_samples, axis=0)
    all_samples = ((all_samples+1.)*(255./2)).astype('int32')
    all_samples = all_samples.reshape((-1, 3, 32, 32)).transpose(0,2,3,1)
    return lib.inception_score.get_inception_score(list(all_samples))

# Dataset iterators
train_gen, dev_gen = lib.cifar10.load(BATCH_SIZE, data_dir=DATA_DIR)
def inf_train_gen():
    while True:
        for images,_ in train_gen():
            yield images

# Train loop
saver = tf.train.Saver(max_to_keep=20)
with tf.Session() as session:
    session.run(tf.initialize_all_variables())
    gen = inf_train_gen()


    p = 0
    max_is = np.zeros((450,1))
    for iteration in xrange(ITERS):
        start_time = time.time()
        # Train generator
        if iteration > 0:
            _ = session.run(gen_train_op)
        # Train critic
        if MODE == 'dcgan':
            disc_iters = 1
        else:
            disc_iters = CRITIC_ITERS
        for i in xrange(disc_iters):
            _data = gen.next()
            _disc_cost, _disc_cost_ad, _add_loss, _ = session.run(
                [disc_cost, disc_cost_ad, ad_loss,disc_train_op],
                feed_dict={real_data_int: _data})
            if MODE == 'wgan':
                _ = session.run(clip_disc_weights)

        lib.plot.plot(checkpoint + '/train disc cost', _disc_cost)
        lib.plot.plot(checkpoint + '/train disc ad cost', _disc_cost_ad)
        lib.plot.plot(checkpoint + '/train ad cost', _add_loss)
        lib.plot.plot(checkpoint + '/time', time.time() - start_time)

        # Calculate inception score every 1K iters
        if iteration % 500 == 499:
            inception_score = get_inception_score()
            max_is[p] = inception_score[0]
            p = p + 1
            print ("max IS:",max(max_is), "max iteration:", max_is.argmax(axis=0))
            #scipy.io.savemat('./result/cifar_stop2_fea/IS.mat', {'IS': max_is})
            scipy.io.savemat(checkpoint + '/IS.mat', {'IS': max_is})
            lib.plot.plot(checkpoint + '/inception score', inception_score[0])

        # Calculate dev loss and generate samples every 100 iters
        if iteration % 100 == 99:
            dev_disc_costs = []
            for images,_ in dev_gen():
                _dev_disc_cost = session.run(disc_cost, feed_dict={real_data_int: images})
                dev_disc_costs.append(_dev_disc_cost)
            lib.plot.plot(checkpoint + '/dev disc cost', np.mean(dev_disc_costs))
            generate_image(iteration, _data)

        # Save logs every 100 iters
        if (iteration < 5) or (iteration % 10000 == 9999):
            saver.save(session, checkpoint + '/checkpoint/save.ckpt', global_step=iteration + 1)
        if (iteration < 5) or (iteration % 100 == 99):
            lib.plot.flush()

        lib.plot.tick()
