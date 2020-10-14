import os
import glob
from PIL import Image
import time
import scipy.misc
import numpy as np
import tensorflow as tf
import model
import tf_slim as slim
tf.compat.v1.disable_eager_execution()
# slim = tf.contrib.slim


class Trainer(object):
    def __init__(self, config):
        filenamequeue = tf.compat.v1.train.string_input_producer([config.filenamequeue])
        self.global_step = tf.Variable(0, name="global_step")
        self.model = self._build_model(filenamequeue, config)
        self.saver = tf.compat.v1.train.Saver(max_to_keep=None)

        if not os.path.exists(config.sampledir):
            os.makedirs(config.sampledir)

        self.loss_summaries = tf.compat.v1.summary.merge([
            tf.compat.v1.summary.scalar("loss_D_real", self.model["loss_D_real"]),
            tf.compat.v1.summary.scalar("loss_D_fake", self.model["loss_D_fake"]),
            tf.compat.v1.summary.scalar("loss_D", self.model["loss_D"]),
            tf.compat.v1.summary.scalar("loss_G", self.model["loss_G"]),
            tf.compat.v1.summary.scalar("loss_E", self.model["loss_E"])])
        self.summary_writer = tf.compat.v1.summary.FileWriter(config.logdir)

        sess_config = tf.compat.v1.ConfigProto(
            allow_soft_placement=True,
            gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
        self.sess = tf.compat.v1.Session(config=sess_config)
        self.config = config


    def _build_model(self, filenamequeue, config):
        x_, y_, tr_, ir_, img_, tex_ = self._im_from_tfrecords(filenamequeue, config)
        z_ = tf.compat.v1.placeholder(tf.float32, [None, config.z_dim], name="z")
        training_ = tf.compat.v1.placeholder(tf.bool, name="is_training")

        category = tf.one_hot(y_, depth=config.y_dim)
        textratio = tf.one_hot(tr_, depth=config.tr_dim)
        imgratio = tf.one_hot(ir_, depth=config.ir_dim)
        x_labeltmp = tf.concat([category, textratio, imgratio], 1)
        var_label, _ = model.embeddingSemvec(x_labeltmp, training_)
        img_fea, _ = model.embeddingImg(img_, training_)
        tex_fea, _ = model.embeddingTex(tex_, training_)

        y_label, _ = model.embeddingFusion(var_label, img_fea, tex_fea, training_)
        ydis_label = tf.reshape(y_label, shape=[-1, 1, 1,
                                config.latent_dim]) * tf.ones([config.batch_size, 64, 64, config.latent_dim])
        encoderdis_label = tf.reshape(y_label, shape=[-1, 1, 1,
                                config.latent_dim]) * tf.ones([config.batch_size, 4, 4, config.latent_dim])
        randomz = tf.random.normal([config.batch_size, config.z_dim])
        # testing case
        testLayout, testImgfea, testSemvec, testTexfea = self.inputs(config)
        randomz_val = np.load('./sample/noiseVector_128.npy')
        testSemvec, _ = model.embeddingSemvec(testSemvec, training_, reuse=True)
        testImgfea, _ = model.embeddingImg(testImgfea, training_, reuse=True)
        testTexfea, _ = model.embeddingTex(testTexfea, training_, reuse=True)
        testlabel, _ = model.embeddingFusion(testSemvec, testImgfea, testTexfea, training_, reuse=True)
        testdislabel = tf.reshape(testlabel, shape=[-1, 1, 1,
                        config.latent_dim]) * tf.ones([128, 4, 4, config.latent_dim])
        test_mean, test_log_sigma_sq, E_end_pts = model.encoder(testLayout, training_, y=testdislabel)
        Etest = test_mean
        Einput = test_mean + tf.exp(test_log_sigma_sq) * randomz_val
        Gtest, Gtest_end_pts = model.generator(Einput, training_, y=testlabel)
        # training case
        z_mean, z_log_sigma_sq, E_end_pts = model.encoder(x_, training_, y=encoderdis_label,
                                                          reuse=True)
        E = z_mean + tf.exp(z_log_sigma_sq) * randomz
        G, G_end_pts = model.generator(z_, training_, y=y_label
                                       , reuse=True)
        Grecon, Grecon_end_pts = model.generator(E, training_,
                                                 y=y_label, reuse=True)
        D_real, D_real_end_pts = model.discriminator(x_, training_,
                                                     y=ydis_label, z=E)
        D_fake, D_fake_end_pts = model.discriminator(G, training_,
                                                     y=ydis_label,
                                                     z=z_, reuse=True)
        # loss
        with tf.compat.v1.variable_scope("Loss_E"):
            kl_div = -0.5 * tf.reduce_sum(input_tensor=1 + 2 * z_log_sigma_sq -
                                          tf.square(z_mean) -
                                          tf.exp(2 * z_log_sigma_sq), axis=1)

            originput = tf.reshape(x_, [config.batch_size, 64 * 64 * 3])
            originput = (originput + 1) / 2.0
            generated_flat = tf.reshape(Grecon,
                                        [config.batch_size, 64 * 64 * 3])
            generated_flat = (generated_flat + 1) / 2.0
            recon_loss = tf.reduce_sum(input_tensor=tf.pow((generated_flat -
                                               originput), 2), axis=1)
            loss_E = tf.reduce_mean(input_tensor=kl_div + recon_loss) / 64 / 64 / 3

        with tf.compat.v1.variable_scope("Loss_D"):
            loss_D_real = tf.reduce_mean(input_tensor=tf.nn.l2_loss(D_real - tf.ones_like(D_real)))
            loss_D_fake = tf.reduce_mean(input_tensor=tf.nn.l2_loss(D_fake - tf.zeros_like(D_fake)))
            loss_D = loss_D_real + loss_D_fake

        with tf.compat.v1.variable_scope("Loss_G"):
            loss_Gls = tf.reduce_mean(input_tensor=tf.nn.l2_loss(D_fake - tf.ones_like(D_fake)))

            kl_div = -0.5 * tf.reduce_sum(input_tensor=1 + 2 * z_log_sigma_sq -
                                          tf.square(z_mean) -
                                          tf.exp(2 * z_log_sigma_sq), axis=1)

            originput = tf.reshape(x_, [config.batch_size, 64 * 64 * 3])
            originput = (originput + 1) / 2.0
            generated_flat = tf.reshape(Grecon,
                                        [config.batch_size, 64 * 64 * 3])
            generated_flat = (generated_flat + 1) / 2.0
            recon_loss = tf.reduce_sum(input_tensor=tf.pow((generated_flat -
                                               originput), 2), axis=1)
            loss_G3 = tf.reduce_mean(input_tensor=kl_div + recon_loss) / 64 / 64 / 3

            recon_loss = tf.reduce_mean(input_tensor=recon_loss) / 64 / 64 / 3
            loss_G = loss_Gls + recon_loss

        with tf.compat.v1.variable_scope("Optimizer_D"):
            vars_D = [var for var in tf.compat.v1.trainable_variables()
                      if "discriminator" in var.name]
            opt_D = tf.compat.v1.train.AdamOptimizer(config.lr,
                                           beta1=config.beta1).minimize(loss_D,
                                                                        self.global_step,
                                                                        var_list=vars_D)

        with tf.compat.v1.variable_scope("Optimizer_G"):
            vars_G = [var for var in tf.compat.v1.trainable_variables()
                      if "generator" in var.name]
            opt_G = tf.compat.v1.train.AdamOptimizer(config.lr,
                                           beta1=config.beta1).minimize(loss_G,
                                                                        var_list=vars_G)
        
        with tf.compat.v1.variable_scope("Optimizer_E"):
            vars_E = [var for var in tf.compat.v1.trainable_variables()
                      if "encoder" in var.name or "embeddingSemvec" in var.name or "embeddingImg" in var.name or "embeddingTex" in var.name or "embeddingFusion" in var.name]
            opt_E = tf.compat.v1.train.AdamOptimizer(config.lr,
                                           beta1=config.beta1).minimize(loss_E,
                                                                        var_list=vars_E)

        return {"x": x_, "y": y_, "z": z_, "is_training": training_,
                "G": G, "E": E,
                "Etest": Etest, "Gtest": Gtest, "Label": testlabel,
                "Grecon": Grecon,
                "D_real": D_real, "D_fake": D_fake,
                "G_end_pts": G_end_pts,
                "D_real_end_pts": D_real_end_pts,
                "D_fake_end_pts": D_fake_end_pts,
                "loss_D_real": loss_D_real, "loss_D_fake": loss_D_fake,
                "loss_D": loss_D, "loss_G": loss_G, "loss_E": loss_E,
                "opt_D": opt_D, "opt_G": opt_G, "opt_E": opt_E}

    def fit(self):
        config = self.config
        self.sess.run(tf.compat.v1.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.compat.v1.train.start_queue_runners(sess=self.sess, coord=coord)

        for step in range(config.max_steps):
            t1 = time.time()
            z = np.random.normal(0.0, 1.0,
                                 size=(config.batch_size,
                                       config.z_dim)).astype(np.float32)
            # train discriminator
            _, d_loss = self.sess.run([self.model["opt_D"],
                                    self.model["loss_D"]],
                                    feed_dict={self.model["z"]: z,
                                                self.model["is_training"]: True})
            # train generator
            _, g_loss = self.sess.run([self.model["opt_G"],
                                    self.model["loss_G"]],
                                    feed_dict={self.model["z"]: z,
                                                self.model["is_training"]: True})
            # train encoder
            _, e_loss = self.sess.run([self.model["opt_E"],
                                       self.model["loss_E"]],
                                      feed_dict={self.model["is_training"]: True})
            while d_loss < 1:
                _, _, g_loss, d_loss, e_loss = self.sess.run([self.model["opt_G"], self.model["opt_E"],
                                                   self.model["loss_G"],
                                                   self.model["loss_D"],
                                                   self.model["loss_E"]],
                                                  feed_dict={self.model["z"]: z,
                                                             self.model["is_training"]: True})
            t2 = time.time()
            if (step + 1) % config.summary_every_n_steps == 0:
                summary_feed_dict = {self.model["z"]: z,
                                     self.model["is_training"]: False}
                print(("step {:5d},loss = (G: {:.8f}, D: {:.8f}), E: {:.8f}"
                       .format(step, g_loss, d_loss, e_loss)))

            if (step + 1) % config.sample_every_n_steps == 0:
                eta = (t2 - t1) * (config.max_steps - step + 1)
                print(("Finished {}/{} step, ETA:{:.2f}s"
                      .format(step + 1, config.max_steps, eta)))
                self.saver.save(self.sess, './log/layoutNet',global_step=(step+1))

                inputdata, gen, fea = self.sample()

                imname = os.path.join(config.sampledir,
                                      "sample" + str(step + 1) + ".jpg")
                h, w = gen.shape[1], gen.shape[2]
                merge_img = np.zeros((h * 16, w * 8, 3))
                for idx, image in enumerate(gen):
                    i = idx % 8
                    j = idx // 8
                    merge_img[j * h:j * h + h, i * w:i * w + w, :] = image
                scipy.misc.imsave(imname, merge_img)
        coord.join(threads)
        sess.close()


    def sample(self):
        inputdata, gen, fea = self.sess.run([self.model["x"],
                                             self.model["Gtest"],
                                             self.model["Etest"]],
                                            feed_dict={self.model["is_training"]: False})
        return (inputdata + 1) / 2.0, (gen + 1) / 2.0, fea

    def testing(self):
        new_saver = tf.compat.v1.train.import_meta_graph('./log/layoutNet-100.meta')
        new_saver.restore(self.sess, './log/layoutNet-100')
        gen, fea = self.sess.run([self.model["Gtest"],
                                  self.model["Etest"]],
                                  feed_dict={self.model["is_training"]: False})

        return (gen + 1) / 2.0, fea

    def inputs(self, config):
        layoutpath = os.getcwd() + '/sample/layout/'
        imgfeapath = os.getcwd() + '/sample/visfea/'
        texfeapath = os.getcwd() + '/sample/texfea/'
        semvecpath = os.getcwd() + '/sample/semvec/'

        f = open('./sample/imgSel_128.txt', 'r')
        name = f.read()
        namelist = name.split()
        n_samples = len(namelist)

        for i in range(n_samples):
            nametmp = namelist[i]
            layoutname = layoutpath + nametmp
            imgfeaname = imgfeapath + nametmp[0:-4] + '.npy'
            semvecname = semvecpath + nametmp[0:-4] + '.npy'
            texfeaname = texfeapath + nametmp[0:-4] + '.npy'

            img1 = Image.open(layoutname)
            rgb = np.array(img1).reshape(1, 64, 64, 3)
            rgb = rgb.astype(np.float32) * (1. / 127.5) - 1.0
            if i == 0:
                testLayout = rgb
            else:
                testLayout = np.concatenate((testLayout, rgb), axis=0)


            imgfea1 = np.load(imgfeaname)
            imgfea1 = imgfea1.reshape((1,14,14,512))
            if i == 0:
                testImgfea = imgfea1
            else:
                testImgfea = np.concatenate((testImgfea, imgfea1), axis=0)
            
            texfea1 = np.load(texfeaname)
            texfea1 = texfea1.reshape((1, 300))
            if i == 0:
                testTexfea = texfea1
            else:
                testTexfea = np.concatenate((testTexfea, texfea1), axis=0)
                        
            convar = np.load(semvecname)
            categoryinput = np.eye(6)[int(convar[0,0])].reshape([1,6])
            textratioinput = np.eye(7)[int(convar[0,1])].reshape([1,7])
            imgratioinput = np.eye(10)[int(convar[0,2])].reshape([1,10])
            semvec1 = np.concatenate([categoryinput, textratioinput, imgratioinput], 1)
            semvec1 = semvec1.astype(np.float32)
            if i == 0:
                testSemvec = semvec1
            else:
                testSemvec = np.concatenate((testSemvec, semvec1), axis=0)

        return testLayout, testImgfea, testSemvec, testTexfea

    def _im_from_tfrecords(self, filenamequeue, config, shuffle=True):
        capacity = config.min_after_dequeue + 3 * config.batch_size
        reader = tf.compat.v1.TFRecordReader()
        _, serialized_example = reader.read(filenamequeue)
        features = tf.io.parse_single_example(
            serialized=serialized_example,
            features={
                "label": tf.io.FixedLenFeature([], tf.int64),
                "textRatio": tf.io.FixedLenFeature([], tf.int64),
                "imgRatio": tf.io.FixedLenFeature([], tf.int64),
                'visualfea': tf.io.FixedLenFeature([], tf.string),
                'textualfea': tf.io.FixedLenFeature([], tf.string),
                "img_raw": tf.io.FixedLenFeature([], tf.string)
            }
        )
        image = tf.cast(tf.reshape((tf.io.decode_raw(features['img_raw'],
                                                  tf.uint8)),
                                   [60, 45, 3]), tf.float32)
        resized_image = tf.image.resize_with_crop_or_pad(image, 64, 64)
        resized_image = resized_image / 127.5 - 1.0
        label = tf.cast(features['label'], tf.int32)
        textRatio = tf.cast(features['textRatio'], tf.int32)
        imgRatio = tf.cast(features['imgRatio'], tf.int32)
        visualfea = features['visualfea']
        visualfea = tf.io.decode_raw(visualfea, tf.float32)
        visualfea = tf.reshape(visualfea, [14, 14, 512])
        textualfea = features['textualfea']
        textualfea = tf.io.decode_raw(textualfea, tf.float32)
        textualfea = tf.reshape(textualfea, [300])

        images, labels, textRatios, imgRatios, visualfea, textualfea = tf.compat.v1.train.shuffle_batch(
            [resized_image, label, textRatio, imgRatio, visualfea, textualfea],
            batch_size=config.batch_size,
            capacity=capacity,
            num_threads=config.num_threads,
            min_after_dequeue=config.min_after_dequeue,
            allow_smaller_final_batch=True,
            name="images")

        return images, labels, textRatios, imgRatios, visualfea, textualfea
