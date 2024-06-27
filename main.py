import os
import os.path as osp
import numpy as np
import random
from datetime import datetime

import paddle
import paddle.nn.functional as F
import paddle.nn as nn
from paddle.vision.transforms import functional as TF
from paddle.io import DataLoader, Dataset
from paddle.static import InputSpec
from paddle.optimizer.lr import StepDecay
from paddle.optimizer import Adam, Momentum
from paddle.regularizer import L2Decay
from paddle.metric import Accuracy
from paddle.fluid.dygraph import to_variable

from model.refinenetlw import rf_lw101
from model.fogpassfilter import FogPassFilter_conv1, FogPassFilter_res1
from utils.losses import CrossEntropy2d
from dataset.paired_cityscapes import Pairedcityscapes
from dataset.Foggy_Zurich import foggyzurichDataSet
from configs.train_config import get_arguments
from utils.optimisers import get_optimisers, get_lr_schedulers

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
RESTORE_FROM = 'without_pretraining'
RESTORE_FROM_fogpass = 'without_pretraining'


def loss_calc(pred, label):
    criterion = CrossEntropy2d()
    return criterion(pred, label)


def gram_matrix(tensor):
    d, h, w = tensor.shape
    tensor = paddle.reshape(tensor, [d, h * w])
    gram = paddle.matmul(tensor, tensor.transpose(perm=[1, 0]))
    return gram


def setup_optimisers_and_schedulers(args, model):
    optimisers = get_optimisers(
        model=model,
        enc_optim_type="momentum",
        enc_lr=6e-4,
        enc_weight_decay=1e-5,
        enc_momentum=0.9,
        dec_optim_type="momentum",
        dec_lr=6e-3,
        dec_weight_decay=1e-5,
        dec_momentum=0.9,
    )
    schedulers = get_lr_schedulers(
        enc_optim=optimisers[0],
        dec_optim=optimisers[1],
        enc_lr_gamma=0.5,
        dec_lr_gamma=0.5,
        enc_scheduler_type="multistep",
        dec_scheduler_type="multistep",
        epochs_per_stage=(100, 100, 100),
    )
    return optimisers, schedulers


def main():
    args = get_arguments()

    paddle.seed(args.random_seed)
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    now = datetime.now().strftime('%m-%d-%H-%M')
    run_name = f'{args.file_name}-{now}'

    wandb.init(project='FIFO', name=f'{run_name}')
    wandb.config.update(args)

    w, h = map(int, args.input_size.split(','))
    input_size = (w, h)

    w_r, h_r = map(int, args.input_size_rf.split(','))
    input_size_rf = (w_r, h_r)

    if args.restore_from == RESTORE_FROM:
        start_iter = 0
        model = rf_lw101(num_classes=args.num_classes)

    else:
        restore = paddle.load(args.restore_from)
        model = rf_lw101(num_classes=args.num_classes)

        model.set_state_dict(restore['state_dict'])
        start_iter = 0

    model.train()
    model.cuda(args.gpu)

    lr_fpf1 = 1e-3
    lr_fpf2 = 1e-3

    if args.modeltrain == 'train':
        lr_fpf1 = 5e-4

    FogPassFilter1 = FogPassFilter_conv1(2080)
    FogPassFilter1_optimizer = Adam(parameters=FogPassFilter1.parameters(), learning_rate=lr_fpf1)
    FogPassFilter1.cuda(args.gpu)

    FogPassFilter2 = FogPassFilter_res1(32896)
    FogPassFilter2_optimizer = Adam(parameters=FogPassFilter2.parameters(), learning_rate=lr_fpf2)
    FogPassFilter2.cuda(args.gpu)

    if args.restore_from_fogpass != RESTORE_FROM_fogpass:
        restore = paddle.load(args.restore_from_fogpass)
        FogPassFilter1.set_state_dict(restore['fogpass1_state_dict'])
        FogPassFilter2.set_state_dict(restore['fogpass2_state_dict'])

    fogpassfilter_loss = paddle.metric.loss.ContrastiveLoss(
        pos_margin=0.1,
        neg_margin=0.1,
        distance=paddle.metric.distance.CosineSimilarity(),
        reducer=paddle.metric.reducer.MeanReducer()
    )

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

    cwsf_pair_loader = DataLoader(
        Pairedcityscapes(
            data_dir=args.data_dir,
            data_dir_cwsf=args.data_dir_cwsf,
            data_list=args.data_list,
            data_list_cwsf=args.data_list_cwsf,
            max_iters=args.num_steps * args.iter_size * args.batch_size,
            mean=IMG_MEAN,
            set=args.set
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True
    )

    rf_loader = DataLoader(
        foggyzurichDataSet(
            data_dir=args.data_dir_rf,
            data_list=args.data_list_rf,
            max_iters=args.num_steps * args.iter_size * args.batch_size,
            mean=IMG_MEAN,
            set=args.set
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True
    )

    cwsf_pair_loader_fogpass = DataLoader(
        Pairedcityscapes(
            data_dir=args.data_dir,
            data_dir_cwsf=args.data_dir_cwsf,
            data_list=args.data_list,
            data_list_cwsf=args.data_list_cwsf,
            max_iters=args.num_steps * args.iter_size * args.batch_size,
            mean=IMG_MEAN,
            set=args.set
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True
    )

    rf_loader_fogpass = DataLoader(
        foggyzurichDataSet(
            data_dir=args.data_dir_rf,
            data_list=args.data_list_rf,
            max_iters=args.num_steps * args.iter_size * args.batch_size,
            mean=IMG_MEAN,
            set=args.set
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True
    )

    rf_loader_iter = enumerate(rf_loader)
    cwsf_pair_loader_iter = enumerate(cwsf_pair_loader)
    cwsf_pair_loader_iter_fogpass = enumerate(cwsf_pair_loader_fogpass)
    rf_loader_iter_fogpass = enumerate(rf_loader_fogpass)

    optimisers, schedulers = setup_optimisers_and_schedulers(args, model=model)
    opts = optimisers if isinstance(optimisers, list) else [optimisers]
    kl_loss = nn.KLDivLoss(reduction='batchmean')
    m = nn.Softmax(axis=1)
    log_m = nn.LogSoftmax(axis=1)

    for i_iter in range(start_iter, args.num_steps):
        loss_seg_cw_value = 0
        loss_seg_sf_value = 0
        loss_fsm_value = 0
        loss_con_value = 0

        for opt in opts:
            opt.clear_grad()

        for sub_i in range(args.iter_size):
            # train fog-pass filtering module
            # freeze the parameters of segmentation network

            model.eval()
            for param in model.parameters():
                param.stop_gradient = True
            for param in FogPassFilter1.parameters():
                param.stop_gradient = False
            for param in FogPassFilter2.parameters():
                param.stop_gradient = False

            _, batch = next(cwsf_pair_loader_iter_fogpass)
            sf_image, cw_image, label, size, sf_name, cw_name = batch

            interp = paddle.nn.functional.interpolate
            _, batch_rf = next(rf_loader_iter_fogpass)
            rf_img, rf_size, rf_name = batch_rf
            img_rf = to_variable(rf_img)
            feature_rf0, feature_rf1, feature_rf2, feature_rf3, feature_rf4, feature_rf5 = model(img_rf)

            images = to_variable(sf_image)
            feature_sf0, feature_sf1, feature_sf2, feature_sf3, feature_sf4, feature_sf5 = model(images)

            images_cw = to_variable(cw_image)
            feature_cw0, feature_cw1, feature_cw2, feature_cw3, feature_cw4, feature_cw5 = model(images_cw)

            fsm_weights = {'layer0': 0.5, 'layer1': 0.5}
            sf_features = {'layer0': feature_sf0, 'layer1': feature_sf1}
            cw_features = {'layer0': feature_cw0, 'layer1': feature_cw1}
            rf_features = {'layer0': feature_rf0, 'layer1': feature_rf1}

            total_fpf_loss = 0

            for idx, layer in enumerate(fsm_weights):
                cw_feature = cw_features[layer]
                sf_feature = sf_features[layer]
                rf_feature = rf_features[layer]
                output, output3 = FogPassFilter1(rf_feature)
                total_fpf_loss += fogpassfilter_loss(output, output3)

                for name, param in FogPassFilter1.named_parameters():
                    if name.startswith("block0"):
                        param.set_lr(6e-4)
                    else:
                        param.set_lr(6e-4)
                for name, param in FogPassFilter1.named_parameters():
                    if name.startswith("block0"):
                        param.set_lr(6e-4)
                    else:
                        param.set_lr(6e-4)

            loss_fsm_value += total_fpf_loss

            for opt in [FogPassFilter1_optimizer]:
                opt.minimize(total_fpf_loss)

        loss_fsm_value /= args.iter_size

        print(f'fogpassfilter Loss: {loss_fsm_value} ')

        model.eval()

        for param in model.parameters():
            param.stop_gradient = False
        for param in FogPassFilter1.parameters():
            param.stop_gradient = True
        for param in FogPassFilter2.parameters():
            param.stop_gradient = True

        _, batch = next(cwsf_pair_loader_iter)

        images, label, size, name = batch

        images = to_variable(images)
        label = to_variable(label)

        pred = model(images)

        loss_seg_cw = loss_calc(pred, label)

        loss_seg_cw.backward()

        for opt in [opts[0]]:
            opt.minimize(loss_seg_cw)

        loss_seg_cw_value += loss_seg_cw.numpy()

        print(f'Segmentation cw loss: {loss_seg_cw_value} ')

        _, batch_rf = next(rf_loader_iter)

        img_rf, rf_size, rf_name = batch_rf

        img_rf = to_variable(img_rf)

        feature_rf0, feature_rf1, feature_rf2, feature_rf3, feature_rf4, feature_rf5 = model(img_rf)

        fogpassfilter_loss = paddle.metric.loss.ContrastiveLoss(
            pos_margin=0.1,
            neg_margin=0.1,
            distance=paddle.metric.distance.CosineSimilarity(),
            reducer=paddle.metric.reducer.MeanReducer()
        )

        total_fpf_loss = 0

        for idx, layer in enumerate(fsm_weights):
            cw_feature = cw_features[layer]
            sf_feature = sf_features[layer]
            rf_feature = rf_features[layer]
            output, output3 = FogPassFilter2(rf_feature)
            total_fpf_loss += fogpassfilter_loss(output, output3)

        loss_fsm_value += total_fpf_loss

        for opt in [FogPassFilter2_optimizer]:
            opt.minimize(total_fpf_loss)

        loss_fsm_value /= args.iter_size

        print(f'fogpassfilter Loss: {loss_fsm_value} ')

        model.eval()

        for param in model.parameters():
            param.stop_gradient = False
        for param in FogPassFilter1.parameters():
            param.stop_gradient = True
        for param in FogPassFilter2.parameters():
            param.stop_gradient = True

        _, batch = next(cwsf_pair_loader_iter)

        images, label, size, name = batch

        images = to_variable(images)
        label = to_variable(label)

        pred = model(images)

        loss_seg_sf = loss_calc(pred, label)

        loss_seg_sf.backward()

        for opt in [opts[1]]:
            opt.minimize(loss_seg_sf)

        loss_seg_sf_value += loss_seg_sf.numpy()

        print(f'Segmentation sf loss: {loss_seg_sf_value} ')

        loss_con_value = loss_seg_sf_value + loss_seg_cw_value

        print(f'loss_con_value: {loss_con_value} ')

        if i_iter >= 0:
            print(f'Loss of {i_iter} iteration, cw seg: {loss_seg_cw_value} seg seg: {loss_seg_sf_value} fog_pass_fi: {loss_fsm_value} ',
                  flush=True)

        # Adjust learning rate
        for scheduler in schedulers:
            scheduler.step()

        if i_iter % args.save_pred_every == 0 and i_iter != 0:
            torch.save(model.state_dict(), osp.join(args.snapshot_dir, f'{i_iter}_{args.file_name}.pth'))
