import json
import os
import pickle as pk
import numpy as np
import torch
from torch.nn.utils import clip_grad
from tqdm import tqdm
from Unet.utils import DataLogger, scale_to_targetsize, visualize_heatmaps
from Unet.metrics.mean_radial_error import mean_radial_error
from Unet.metrics.successful_detection_rate import successful_detection_rate


def clip_gradient(optimizer, max_norm, norm_type):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group["params"]:
            clip_grad.clip_grad_norm_(param, max_norm, norm_type)

# forward pass, loss & acc calculation, 
def train(opt, cfg, train_loader, m, criterion, optimizer):
    # initialize the logger for loss and accuracy
    # here cfg as the whole configuration
    loss_logger = DataLogger()

    m.train() # set model to be in training mode
    
    grad_clip = cfg.TRAIN.get('GRAD_CLIP', False) # default as False

    # using tqdm bar
    if opt.log:
        train_loader = tqdm(train_loader, dynamic_ncols=True)

    # enumerate over the training data
    for i, (inps, labels, _) in enumerate(train_loader):
        inps = inps.cuda(opt.gpu) # put data to GPU, image for cephalometric dataset

        for k, _ in labels.items():
            if k == 'type':
                continue

            labels[k] = labels[k].cuda(opt.gpu) # put labels to GPU, I delete the opt.gpu
        # print(len(labels))
        output = m(inps) # forward pass

        
        loss = criterion(output, labels) # loss calculation, type's default is 'heatmap' with another choice is 'coord'

        if isinstance(inps, list):
            batch_size = inps[0].size(0)
        else:
            batch_size = inps.size(0)

        # update the logger for loss and accuracy
        loss_logger.update(loss.item(), batch_size)

        # backward pass and optimization step
        optimizer.zero_grad()
        loss.backward()

        if grad_clip:
            clip_gradient(optimizer, grad_clip.MAX_NORM, grad_clip.NORM_TYPE)
        optimizer.step()

        opt.trainIters += 1 # update the trainIters

        if opt.log:
            # TQDM: display the progress,while in validate, the final result is the key
            train_loader.set_description(
                'loss: {loss:.8f}'.format(
                    loss=loss_logger.avg)
            )

    if opt.log:
        train_loader.close()

    return loss_logger.avg # return the average loss and accuracy after one epoch


def validate(opt, cfg, val_loader, m, heatmap_to_coord, show_hm=False):
    
    kpt_json = []   # store the validation keypoints
    mre_logger = DataLogger()  # store the mean radial error
    sd_logger = DataLogger() # store the standard deviation of radial error
    sdr_logger = DataLogger() # store the successful detection rate
    
    m.eval()    # model to be in evaluation mode

    subset_str = val_loader.dataset.subset # get the subset string
    if opt.log:
        val_loader = tqdm(val_loader, dynamic_ncols=True)

    batchNumVisual = 1  # number of batches for images to visualize
    batchCnt = 0
    # enumerate over the validation data
    for inps, labels, img_ids in val_loader:
        BATCHSIZE = inps.size(0)
        # data put to GPU
        inps = inps.cuda(opt.gpu)
        for k, _ in labels.items():
            if k == 'type':
                continue

            labels[k] = labels[k].cuda(opt.gpu)

        output = m(inps) # forward pass, output has two items, one for 'heatmap' and one for 'pred_pts'
        # output['heatmap']:  [batch_size, num_keypoints, hm_height, hm_width]
        # output['pred_pts']: [batch_size, num_keypoints, 2] of HEATMAP_SIZE
        # labels['target_hm']: [batch_size, num_keypoints, hm_height, hm_width]
        # labels['target_uv']: [batch_size, num_keypoints * 2]

        if show_hm and batchCnt < batchNumVisual:
            # 只看一批
            print("Heatmap Visualization is doing, please wait... ")
            visualize_heatmaps(opt, inps, labels['target_hm'], output['heatmap'], img_ids, target_uv=labels['target_uv'].reshape(BATCHSIZE,-1, 2), pred_uv=output['pred_pts'], type=cfg.DATASET.PRESET.METHOD_TYPE, show_norm=cfg.VALIDATE.SHOWNORM)
                # here the target_uv and pred_uv is [-0.5, 0.5]
            batchCnt += 1

        # convert the output['pred_pts'] from HEATMAP_SIZE to rawImage_SIZE
        # 1. convert from [-0.5, 0.5] to HEATMAP_SIZE
        # 2. map from IMAGE_SIZE to rawImage_SIZE

        from Unet.utils import get_center_scale, transform_preds

        HEATMAP_SIZE_H, HEATMAP_SIZE_W = cfg.DATASET.PRESET.HEATMAP_SIZE
        rawImage_SIZE_H, rawImage_SIZE_W = cfg.DATASET.PRESET.RAW_IMAGE_SIZE
        target_ratio_w_h = HEATMAP_SIZE_W / HEATMAP_SIZE_H
        center, scale = get_center_scale(rawImage_SIZE_W, rawImage_SIZE_H, target_ratio_w_h, scale_mult=1.25)

        output['pred_pts'][:, :, 0] = (output['pred_pts'][:, :, 0] + 0.5) * HEATMAP_SIZE_H
        output['pred_pts'][:, :, 1] = (output['pred_pts'][:, :, 1] + 0.5) * HEATMAP_SIZE_W
        output['pred_pts'].cpu().numpy()
        
        pred_pts_rawImage_SIZE = np.zeros((BATCHSIZE, cfg.DATASET.PRESET.NUM_JOINTS, 2))  # [batch_size, num_keypoints, 2] of rawImage_SIZE
        for i in range(BATCHSIZE):
            for j in range(cfg.DATASET.PRESET.NUM_JOINTS):
                pred_pts_rawImage_SIZE[i, j, :] = transform_preds(output['pred_pts'][i, j, :].cpu().numpy(), center, scale, [HEATMAP_SIZE_W, HEATMAP_SIZE_H]) # 一个x,y坐标对



        # convert labels['target_uv'] to rawImage_SIZE
        # 1. convert from [-0.5, 0.5] to IMAGE_SIZE
        # 2. map from IMAGE_SIZE to rawImage_SIZE
        IMAGE_SIZE_H, IMAGE_SIZE_W = cfg.DATASET.PRESET.IMAGE_SIZE
        labels['target_uv'] = scale_to_targetsize(labels['target_uv'], cfg.DATASET.PRESET.IMAGE_SIZE) # [batch_size, num_keypoints * 2]
        labels['target_uv'] = labels['target_uv'].reshape(BATCHSIZE, -1, 2) # [batch_size, num_keypoints, 2]
        gt_pts_rawImage_SIZE = np.zeros((BATCHSIZE, cfg.DATASET.PRESET.NUM_JOINTS, 2))  # [batch_size, num_keypoints, 2] of rawImage_SIZE

        for i in range(BATCHSIZE):
            for j in range(cfg.DATASET.PRESET.NUM_JOINTS):
                gt_pts_rawImage_SIZE[i, j, :] = transform_preds(labels['target_uv'][i, j, :].cpu().numpy(), center, scale, [IMAGE_SIZE_W, IMAGE_SIZE_H]) # 一个x,y坐标对

        # convert heatmap to coordinates and save to json file

        


        # convert heatmap to coordinates and save to json file
        for i in range(inps.shape[0]):
            #pose_coords, pose_scores = heatmap_to_coord(output, idx=i) # pose_coords [N, num_joint, 2] corresponding hm_size
            data = dict()
            data['image_id'] = str(img_ids[i])
            #data['pred_hm'] = output['heatmap'][i].detach().cpu().tolist()
            #data['target_hm'] = labels['target_hm'][i].detach().cpu().tolist()

            pred_pts = pred_pts_rawImage_SIZE[i].reshape(-1)  # [num_keypoints * 2]
            data['pred_pts'] = pred_pts.tolist()

            targets_pts = gt_pts_rawImage_SIZE[i].reshape(-1)  # [num_keypoints * 2]
            data['target_pts'] = targets_pts.tolist()
            kpt_json.append(data)

        # calculate the mean radial error and successful detection rate, the input here is of dimension [batchsize, num_joints * 2]
        mre, sd = mean_radial_error(pred_pts_rawImage_SIZE, gt_pts_rawImage_SIZE, target_size=cfg.DATASET.PRESET.RAW_IMAGE_SIZE)
        sdr = successful_detection_rate(pred_pts_rawImage_SIZE, gt_pts_rawImage_SIZE, target_size=cfg.DATASET.PRESET.RAW_IMAGE_SIZE, radii=cfg.VALIDATE.RADII)

        if isinstance(inps, list):
            batch_size = inps[0].size(0)
        else:
            batch_size = inps.size(0)

        mre_logger.update(mre, batch_size)
        sd_logger.update(sd, batch_size)
        sdr_logger.update(sdr, batch_size)

    if opt.log:
        val_loader.close()
        
    # save the results
    with open(os.path.join(opt.work_dir, f'{subset_str}_gt_kpt.json'), 'w') as fid:
        json.dump(kpt_json, fid)

        
    return mre_logger.avg, sd_logger.avg, sdr_logger.avg







