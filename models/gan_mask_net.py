import numpy as np

# Theano
import theano
import theano.tensor as tensor

from models.net import Net, tensor5
from lib.config import cfg
from lib.layers import TensorProductLayer, ConvLayer, PoolLayer, Unpool3DLayer, \
    LeakyReLU, SoftmaxWithLoss3D, Conv3DLayer, InputLayer, FlattenLayer, \
    FCConv3DLayer, TanhLayer, SigmoidLayer, ComplementLayer, AddLayer, \
    EltwiseMultiplyLayer, RaytracingLayer, DimShuffleLayer, Pool3DLayer, \
    DifferentiableStepLayer, SubtractLayer, InstanceNoiseLayer, \
    get_trainable_params


class GANMaskNet(Net):

    def network_definition(self):

        # (multi_views, self.batch_size, 3, self.img_h, self.img_w),
        self.x = tensor5()
        self.is_x_tensor4 = False

        img_w = self.img_w
        img_h = self.img_h
        n_gru_vox = 4
        n_vox = self.n_vox

        n_convfilter = [96, 128, 256, 256, 256, 256]
        n_fc_filters = [1024, 2]
        n_deconvfilter = [128, 128, 128, 128, 96, 2]
        n_conv_advfilter = [32, 128, 128, 128, 32]
        n_fc_advfilter = [1024, 2]
        input_shape = (self.batch_size, 3, img_w, img_h)
        voxel_shape = (self.batch_size, n_vox, n_vox, n_vox)

        # To define weights, define the network structure first
        x = InputLayer(input_shape)
        conv1a = ConvLayer(x, (n_convfilter[0], 7, 7), param_type='generator')
        conv1b = ConvLayer(conv1a, (n_convfilter[0], 3, 3), param_type='generator')
        pool1 = PoolLayer(conv1b)

        conv2a = ConvLayer(pool1, (n_convfilter[1], 3, 3), param_type='generator')
        conv2b = ConvLayer(conv2a, (n_convfilter[1], 3, 3), param_type='generator')
        conv2c = ConvLayer(pool1, (n_convfilter[1], 1, 1), param_type='generator')
        pool2 = PoolLayer(conv2c)

        conv3a = ConvLayer(pool2, (n_convfilter[2], 3, 3), param_type='generator')
        conv3b = ConvLayer(conv3a, (n_convfilter[2], 3, 3), param_type='generator')
        conv3c = ConvLayer(pool2, (n_convfilter[2], 1, 1), param_type='generator')
        pool3 = PoolLayer(conv3b)

        conv4a = ConvLayer(pool3, (n_convfilter[3], 3, 3), param_type='generator')
        conv4b = ConvLayer(conv4a, (n_convfilter[3], 3, 3), param_type='generator')
        pool4 = PoolLayer(conv4b)

        conv5a = ConvLayer(pool4, (n_convfilter[4], 3, 3), param_type='generator')
        conv5b = ConvLayer(conv5a, (n_convfilter[4], 3, 3), param_type='generator')
        conv5c = ConvLayer(pool4, (n_convfilter[4], 1, 1), param_type='generator')
        pool5 = PoolLayer(conv5b)

        conv6a = ConvLayer(pool5, (n_convfilter[5], 3, 3), param_type='generator')
        conv6b = ConvLayer(conv6a, (n_convfilter[5], 3, 3), param_type='generator')
        pool6 = PoolLayer(conv6b)

        flat6 = FlattenLayer(pool6)
        fc7 = TensorProductLayer(flat6, n_fc_filters[0], param_type='generator')

        # Set the size to be 256x4x4x4
        s_shape = (self.batch_size, n_gru_vox, n_deconvfilter[0], n_gru_vox, n_gru_vox)

        # Dummy 3D grid hidden representations
        prev_s = InputLayer(s_shape)

        t_x_s_update = FCConv3DLayer(prev_s, fc7, (n_deconvfilter[0], n_deconvfilter[0], 3, 3, 3), param_type='generator')
        t_x_s_reset = FCConv3DLayer(prev_s, fc7, (n_deconvfilter[0], n_deconvfilter[0], 3, 3, 3), param_type='generator')

        reset_gate = SigmoidLayer(t_x_s_reset)

        rs = EltwiseMultiplyLayer(reset_gate, prev_s)
        t_x_rs = FCConv3DLayer(rs, fc7, (n_deconvfilter[0], n_deconvfilter[0], 3, 3, 3), param_type='generator')

        def recurrence(x_curr, prev_s_tensor, prev_in_gate_tensor):
            # Scan function cannot use compiled function.
            input_ = InputLayer(input_shape, x_curr)
            conv1a_ = ConvLayer(input_, (n_convfilter[0], 7, 7), params=conv1a.params)
            rect1a_ = LeakyReLU(conv1a_)
            conv1b_ = ConvLayer(rect1a_, (n_convfilter[0], 3, 3), params=conv1b.params)
            rect1_ = LeakyReLU(conv1b_)
            pool1_ = PoolLayer(rect1_)

            conv2a_ = ConvLayer(pool1_, (n_convfilter[1], 3, 3), params=conv2a.params)
            rect2a_ = LeakyReLU(conv2a_)
            conv2b_ = ConvLayer(rect2a_, (n_convfilter[1], 3, 3), params=conv2b.params)
            rect2_ = LeakyReLU(conv2b_)
            conv2c_ = ConvLayer(pool1_, (n_convfilter[1], 1, 1), params=conv2c.params)
            res2_ = AddLayer(conv2c_, rect2_)
            pool2_ = PoolLayer(res2_)

            conv3a_ = ConvLayer(pool2_, (n_convfilter[2], 3, 3), params=conv3a.params)
            rect3a_ = LeakyReLU(conv3a_)
            conv3b_ = ConvLayer(rect3a_, (n_convfilter[2], 3, 3), params=conv3b.params)
            rect3_ = LeakyReLU(conv3b_)
            conv3c_ = ConvLayer(pool2_, (n_convfilter[2], 1, 1), params=conv3c.params)
            res3_ = AddLayer(conv3c_, rect3_)
            pool3_ = PoolLayer(res3_)

            conv4a_ = ConvLayer(pool3_, (n_convfilter[3], 3, 3), params=conv4a.params)
            rect4a_ = LeakyReLU(conv4a_)
            conv4b_ = ConvLayer(rect4a_, (n_convfilter[3], 3, 3), params=conv4b.params)
            rect4_ = LeakyReLU(conv4b_)
            pool4_ = PoolLayer(rect4_)

            conv5a_ = ConvLayer(pool4_, (n_convfilter[4], 3, 3), params=conv5a.params)
            rect5a_ = LeakyReLU(conv5a_)
            conv5b_ = ConvLayer(rect5a_, (n_convfilter[4], 3, 3), params=conv5b.params)
            rect5_ = LeakyReLU(conv5b_)
            conv5c_ = ConvLayer(pool4_, (n_convfilter[4], 1, 1), params=conv5c.params)
            res5_ = AddLayer(conv5c_, rect5_)
            pool5_ = PoolLayer(res5_)

            conv6a_ = ConvLayer(pool5_, (n_convfilter[5], 3, 3), params=conv6a.params)
            rect6a_ = LeakyReLU(conv6a_)
            conv6b_ = ConvLayer(rect6a_, (n_convfilter[5], 3, 3), params=conv6b.params)
            rect6_ = LeakyReLU(conv6b_)
            res6_ = AddLayer(pool5_, rect6_)
            pool6_ = PoolLayer(res6_)

            flat6_ = FlattenLayer(pool6_)
            fc7_ = TensorProductLayer(flat6_, n_fc_filters[0], params=fc7.params)
            rect7_ = LeakyReLU(fc7_)

            prev_s_ = InputLayer(s_shape, prev_s_tensor)

            t_x_s_update_ = FCConv3DLayer(
                prev_s_,
                rect7_, (n_deconvfilter[0], n_deconvfilter[0], 3, 3, 3),
                params=t_x_s_update.params)

            t_x_s_reset_ = FCConv3DLayer(
                prev_s_,
                rect7_, (n_deconvfilter[0], n_deconvfilter[0], 3, 3, 3),
                params=t_x_s_reset.params)

            update_gate_ = SigmoidLayer(t_x_s_update_)
            comp_update_gate_ = ComplementLayer(update_gate_)
            reset_gate_ = SigmoidLayer(t_x_s_reset_)

            rs_ = EltwiseMultiplyLayer(reset_gate_, prev_s_)
            t_x_rs_ = FCConv3DLayer(
                rs_, rect7_, (n_deconvfilter[0], n_deconvfilter[0], 3, 3, 3), params=t_x_rs.params)
            tanh_t_x_rs_ = TanhLayer(t_x_rs_)

            gru_out_ = AddLayer(
                EltwiseMultiplyLayer(update_gate_, prev_s_),
                EltwiseMultiplyLayer(comp_update_gate_, tanh_t_x_rs_))

            return gru_out_.output, update_gate_.output

        s_update, r_update = theano.scan(recurrence,
                sequences=[self.x[:, :, :3]],  # along with images, feed in the index of the current frame
            outputs_info=[tensor.zeros_like(np.zeros(s_shape),
                                            dtype=theano.config.floatX),
                           tensor.zeros_like(np.zeros(s_shape),
                                             dtype=theano.config.floatX)])

        update_all = s_update[-1]
        s_all = s_update[0]
        s_last = s_all[-1]
        gru_s   = InputLayer(s_shape, s_last)
        unpool7 = Unpool3DLayer(gru_s)
        conv7a  = Conv3DLayer(unpool7, (n_deconvfilter[1], 3, 3, 3),
                param_type='generator')
        rect7a  = LeakyReLU(conv7a)
        conv7b  = Conv3DLayer(rect7a, (n_deconvfilter[1], 3, 3, 3),
                param_type='generator')
        rect7   = LeakyReLU(conv7b)
        res7    = AddLayer(unpool7, rect7)

        unpool8 = Unpool3DLayer(res7)
        conv8a  = Conv3DLayer(unpool8, (n_deconvfilter[2], 3, 3, 3),
                param_type='generator')
        rect8a  = LeakyReLU(conv8a)
        conv8b  = Conv3DLayer(rect8a, (n_deconvfilter[2], 3, 3, 3),
                param_type='generator')
        rect8   = LeakyReLU(conv8b)
        res8    = AddLayer(unpool8, rect8)

        unpool9 = Unpool3DLayer(res8)
        conv9a  = Conv3DLayer(unpool9, (n_deconvfilter[3], 3, 3, 3),
                param_type='generator')
        rect9a  = LeakyReLU(conv9a)
        conv9b  = Conv3DLayer(rect9a, (n_deconvfilter[3], 3, 3, 3),
                param_type='generator')
        rect9   = LeakyReLU(conv9b)

        conv9c  = Conv3DLayer(unpool9, (n_deconvfilter[3], 1, 1, 1),
                param_type='generator')
        res9    = AddLayer(conv9c, rect9)

        conv10a = Conv3DLayer(res9, (n_deconvfilter[3], 3, 3, 3),
                param_type='generator')
        rect10a = LeakyReLU(conv10a)
        conv10b = Conv3DLayer(rect10a, (n_deconvfilter[3], 3, 3, 3),
                param_type='generator')
        rect10  = LeakyReLU(conv10b)

        conv10c = Conv3DLayer(rect10a, (n_deconvfilter[3], 3, 3, 3),
                param_type='generator')
        res10   = AddLayer(conv10c, rect10)

        conv11  = Conv3DLayer(res10, (n_deconvfilter[4], 3, 3, 3),
                param_type='generator')
        conv12  = Conv3DLayer(conv11, (n_deconvfilter[5], 3, 3, 3),
                param_type='generator')
        voxel_loss = SoftmaxWithLoss3D(conv12.output)
        reconstruction = voxel_loss.prediction()

        voxel_input = InputLayer(voxel_shape, reconstruction[:, :, 1])
        rend = RaytracingLayer(voxel_input, self.camera, img_w, img_h,
                               self.pad_x, self.pad_y)

        # Discriminator network starts here.
        disc_input = InputLayer(voxel_shape)
        disc_padded = DimShuffleLayer(disc_input, (0, 1, 'x', 2, 3))
        conv15  = Conv3DLayer(disc_padded, (n_conv_advfilter[0], 3, 3, 3),
                param_type='discriminator')
        conv16  = Conv3DLayer(conv15, (n_conv_advfilter[0], 3, 3, 3),
                param_type='discriminator')
        pool16 = Pool3DLayer(conv16)  # b x 16 x c x 16 x 16
        conv17  = Conv3DLayer(pool16, (n_conv_advfilter[1], 3, 3, 3),
                param_type='discriminator')
        conv18  = Conv3DLayer(conv17, (n_conv_advfilter[1], 3, 3, 3),
                param_type='discriminator')
        pool18 = Pool3DLayer(conv18)  # b x 8 x c x 8 x 8
        conv19  = Conv3DLayer(pool18, (n_conv_advfilter[2], 3, 3, 3),
                param_type='discriminator')
        conv20  = Conv3DLayer(conv19, (n_conv_advfilter[2], 3, 3, 3),
                param_type='discriminator')
        pool20 = Pool3DLayer(conv20)  # b x 4 x c x 4 x 4
        conv21  = Conv3DLayer(pool20, (n_conv_advfilter[3], 3, 3, 3),
                param_type='discriminator')
        conv22  = Conv3DLayer(conv21, (n_conv_advfilter[3], 3, 3, 3),
                param_type='discriminator')
        pool22 = Pool3DLayer(conv22)  # b x 2 x c x 2 x 2
        conv23  = Conv3DLayer(pool22, (n_conv_advfilter[4], 3, 3, 3),
                param_type='discriminator')
        conv24  = Conv3DLayer(conv23, (n_conv_advfilter[4], 1, 1, 1),
                param_type='discriminator')
        flat24 = FlattenLayer(conv24)
        fc24 = TensorProductLayer(flat24, n_fc_advfilter[1],
                param_type='discriminator')

        def get_discriminator(data_centered, use_dropout):
            conv15_  = Conv3DLayer(data_centered,
                    (n_conv_advfilter[0], 3, 3, 3), params=conv15.params)
            rect15_ = LeakyReLU(conv15_)
            conv16_  = Conv3DLayer(rect15_, (n_conv_advfilter[0], 3, 3, 3),
                    params=conv16.params)
            rect16_ = LeakyReLU(conv16_)
            pool16_ = Pool3DLayer(rect16_)  # b x 16 x c x 16 x 16
            conv17_  = Conv3DLayer(pool16_, (n_conv_advfilter[1], 3, 3, 3),
                    params=conv17.params)
            rect17_ = LeakyReLU(conv17_)
            conv18_  = Conv3DLayer(rect17_, (n_conv_advfilter[1], 3, 3, 3),
                    params=conv18.params)
            rect18_ = LeakyReLU(conv18_)
            pool18_ = Pool3DLayer(rect18_)  # b x 8 x c x 8 x 8
            conv19_  = Conv3DLayer(pool18_, (n_conv_advfilter[2], 3, 3, 3),
                    params=conv19.params)
            rect19_ = LeakyReLU(conv19_)
            conv20_  = Conv3DLayer(rect19_, (n_conv_advfilter[2], 3, 3, 3),
                    params=conv20.params)
            rect20_ = LeakyReLU(conv20_)
            pool20_ = Pool3DLayer(rect20_)  # b x 4 x c x 4 x 4
            conv21_  = Conv3DLayer(pool20_, (n_conv_advfilter[3], 3, 3, 3),
                    params=conv21.params)
            rect21_ = LeakyReLU(conv21_)
            conv22_  = Conv3DLayer(rect21_, (n_conv_advfilter[3], 3, 3, 3),
                    params=conv22.params)
            rect22_ = LeakyReLU(conv22_)
            pool22_ = Pool3DLayer(rect22_)  # b x 2 x c x 2 x 2
            conv23_  = Conv3DLayer(pool22_, (n_conv_advfilter[4], 3, 3, 3),
                    params=conv23.params)
            rect23_ = LeakyReLU(conv23_)
            conv24_  = Conv3DLayer(rect23_, (n_conv_advfilter[4], 1, 1, 1),
                    params=conv24.params)
            flat24_ = FlattenLayer(conv24_)
            fc24_ = TensorProductLayer(flat24_, n_fc_advfilter[1],
                    params=fc24.params)
            return SoftmaxWithLoss3D(fc24_.output, axis=1)

        voxel_padded = DimShuffleLayer(voxel_input, (0, 1, 'x', 2, 3))
        if cfg.TRAIN.STABILIZER == 'diffstep':
            voxel_stabilized = DifferentiableStepLayer(voxel_padded,
                    backprop=cfg.TRAIN.DIFF_BACKPROP)
        elif cfg.TRAIN.STABILIZER == 'noise':
            voxel_stabilized = InstanceNoiseLayer(voxel_padded,
                    std=self.noise * cfg.TRAIN.NOISE_MAXSTD)
        elif cfg.TRAIN.STABILIZER == 'ignore':
            voxel_stabilized = voxel_padded
        else:
            raise NotImplemented
        voxel_centered = SubtractLayer(voxel_stabilized, 0.5)

        gt_input = InputLayer(voxel_shape, self.y[:, :, 1])
        gt_padded = DimShuffleLayer(gt_input, (0, 1, 'x', 2, 3))
        if cfg.TRAIN.STABILIZER == 'diffstep':
            gt_stabilized = gt_padded
        elif cfg.TRAIN.STABILIZER == 'noise':
            gt_stabilized = InstanceNoiseLayer(gt_padded,
                    std=self.noise * cfg.TRAIN.NOISE_MAXSTD)
        elif cfg.TRAIN.STABILIZER == 'ignore':
            gt_stabilized = gt_padded
        else:
            raise NotImplemented
        gt_centered = SubtractLayer(gt_stabilized, 0.5)

        # Discriminator 1: takes fake voxel as input.
        discriminator_fake_loss = get_discriminator(voxel_centered, True)

        # Discriminator 2: takes real voxel as input.
        discriminator_real_loss = get_discriminator(gt_centered, True)

        # Discriminator 3: takes generated voxel as input, doesn't use dropout.
        discriminator_fake_test = get_discriminator(voxel_centered, False)

        # Discriminator 4: takes real voxel as input, doesn't use dropout.
        discriminator_real_test = get_discriminator(gt_centered, False)

        assert not r_update, 'Unexpected update in the RNN.'
        label_shape = np.zeros((self.batch_size, 1))
        fake_label = tensor.zeros_like(label_shape, dtype=theano.config.floatX)
        real_label = tensor.ones_like(label_shape, dtype=theano.config.floatX)
        all_fake = tensor.concatenate((real_label, fake_label), axis=1)
        all_real = tensor.concatenate((fake_label, real_label), axis=1)
        self.voxel_loss = discriminator_fake_test.loss(all_real)
        self.mask_loss = tensor.nnet.nnet.binary_crossentropy(
                tensor.clip(rend.output[:, :, 0], 1e-7, 1.0 - 1e-7),
                tensor.gt(self.x[:, :, 3], 0.).astype(theano.config.floatX)).mean()
        self.discriminator_loss = (discriminator_fake_loss.loss(all_fake) +
                                   discriminator_real_loss.loss(all_real)) / 2.
        self.generator_loss = self.voxel_loss + self.mask_loss * 100
        self.error = voxel_loss.error(self.y)
        self.error_F = discriminator_fake_test.error(all_fake)
        self.error_R = discriminator_real_test.error(all_real)
        self.generator_params = get_trainable_params()['generator']
        self.discriminator_params = get_trainable_params()['discriminator']
        self.all_params = self.generator_params + self.discriminator_params
        self.load_params = self.all_params
        self.output = reconstruction
        self.activations = [rend.output[:, :, 0]]
