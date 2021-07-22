import os
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))
import time
import argparse
import os
import cv2
import glob
import copy
import numpy as np
import torch
import imageio
from PIL import Image
import scipy.ndimage
from skimage.feature import canny
import torchvision.transforms.functional as F

from RAFT import utils
from RAFT import RAFT

import utils.region_fill as rf
from utils.Poisson_blend import Poisson_blend
from utils.Poisson_blend_img import Poisson_blend_img
from get_flowNN import get_flowNN
from get_flowNN_gradient import get_flowNN_gradient
from utils.common_utils import flow_edge
from spatial_inpaint import spatial_inpaint
from frame_inpaint import DeepFillv1
from edgeconnect.networks import EdgeGenerator_
from torch.multiprocessing import Queue


def to_tensor(img):
    img = Image.fromarray(img)
    img_t = F.to_tensor(img).float()
    return img_t


def infer(args, EdgeGenerator, device, flow_img_gray, edge, mask):

    # Add a pytorch dataloader
    flow_img_gray_tensor = to_tensor(flow_img_gray)[None, :, :].float().to(device)
    edge_tensor = to_tensor(edge)[None, :, :].float().to(device)
    mask_tensor = torch.from_numpy(mask.astype(np.float64))[None, None, :, :].float().to(device)

    # Complete the edges
    edges_masked = (edge_tensor * (1 - mask_tensor))
    images_masked = (flow_img_gray_tensor * (1 - mask_tensor)) + mask_tensor
    inputs = torch.cat((images_masked, edges_masked, mask_tensor), dim=1)
    with torch.no_grad():
        edges_completed = EdgeGenerator(inputs) # in: [grayscale(1) + edge(1) + mask(1)]
    edges_completed = edges_completed * mask_tensor + edge_tensor * (1 - mask_tensor)
    edge_completed = edges_completed[0, 0].data.cpu().numpy()
    edge_completed[edge_completed < 0.5] = 0
    edge_completed[edge_completed >= 0.5] = 1

    return edge_completed


def gradient_mask(mask):

    gradient_mask = np.logical_or.reduce((mask,
        np.concatenate((mask[1:, :], np.zeros((1, mask.shape[1]), dtype=np.bool)), axis=0),
        np.concatenate((mask[:, 1:], np.zeros((mask.shape[0], 1), dtype=np.bool)), axis=1)))

    return gradient_mask


def create_dir(dir):
    """Creates a directory if not exist.
    """
    if not os.path.exists(dir):
        os.makedirs(dir)


def initialize_RAFT(args):
    """Initializes the RAFT model.
    """
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to('cuda')
    model.eval()
    return model


def calculate_flow(args, model, video, mode):
    """Calculates optical flow.
    """
    if mode not in ['forward', 'backward']:
        raise NotImplementedError

    nFrame, _, imgH, imgW = video.shape
    Flow = np.empty(((imgH, imgW, 2, 0)), dtype=np.float32)

    # if os.path.isdir(os.path.join(args.outroot, 'flow', mode + '_flo')):
    #     for flow_name in sorted(glob.glob(os.path.join(args.outroot, 'flow', mode + '_flo', '*.flo'))):
    #         print("Loading {0}".format(flow_name), '\r', end='')
    #         flow = utils.frame_utils.readFlow(flow_name)
    #         Flow = np.concatenate((Flow, flow[..., None]), axis=-1)
    #     return Flow

    # create_dir(os.path.join('testImg','flow', mode + '_flo'))
    # create_dir(os.path.join('testImg','flow', mode + '_png'))

    with torch.no_grad():
        for i in range(video.shape[0] - 1):
            print("Calculating {0} flow {1:2d} <---> {2:2d}".format(mode, i, i + 1), '\r', end='')
            if mode == 'forward':
                # Flow i -> i + 1
                image1 = video[i, None]
                image2 = video[i + 1, None]
            elif mode == 'backward':
                # Flow i + 1 -> i
                image1 = video[i + 1, None]
                image2 = video[i, None]
            else:
                raise NotImplementedError

            _, flow = model(image1, image2, iters=20, test_mode=True)
            flow = flow[0].permute(1, 2, 0).cpu().numpy()
            Flow = np.concatenate((Flow, flow[..., None]), axis=-1)

            # Flow visualization.
            flow_img = utils.flow_viz.flow_to_image(flow)
            flow_img = Image.fromarray(flow_img)

            # Saves the flow and flow_img.
            # flow_img.save(os.path.join('testImg','flow', mode + '_png', '%05d.png'%i))
            # utils.frame_utils.writeFlow(os.path.join('testImg','flow', mode + '_flo', '%05d.flo'%i), flow)

    return Flow
def complete_flow(args, corrFlow, flow_mask, mode, edge=None):
    """Completes flow.
    """
    if mode not in ['forward', 'backward']:
        raise NotImplementedError

    imgH, imgW, _, nFrame = corrFlow.shape

    # if os.path.isdir(os.path.join(args.outroot, 'flow_comp', mode + '_flo')):
    #     compFlow = np.empty(((imgH, imgW, 2, 0)), dtype=np.float32)
    #     for flow_name in sorted(glob.glob(os.path.join(args.outroot, 'flow_comp', mode + '_flo', '*.flo'))):
    #         print("Loading {0}".format(flow_name), '\r', end='')
    #         flow = utils.frame_utils.readFlow(flow_name)
    #         compFlow = np.concatenate((compFlow, flow[..., None]), axis=-1)
    #     return compFlow

    # create_dir(os.path.join('testImg','flow_comp', mode + '_flo'))
    # create_dir(os.path.join('testImg','flow_comp', mode + '_png'))

    compFlow = np.zeros(((imgH, imgW, 2, nFrame)), dtype=np.float32)

    for i in range(nFrame):
        print("Completing {0} flow {1:2d} <---> {2:2d}".format(mode, i, i + 1), '\r', end='')
        flow = corrFlow[:, :, :, i]
        flow_mask_img = flow_mask[:, :, i] if mode == 'forward' else flow_mask[:, :, i + 1]
        flow_mask_gradient_img = gradient_mask(flow_mask_img)

        if edge is not None:
            # imgH x (imgW - 1 + 1) x 2
            gradient_x = np.concatenate((np.diff(flow, axis=1), np.zeros((imgH, 1, 2), dtype=np.float32)), axis=1)
            # (imgH - 1 + 1) x imgW x 2
            gradient_y = np.concatenate((np.diff(flow, axis=0), np.zeros((1, imgW, 2), dtype=np.float32)), axis=0)

            # concatenate gradient_x and gradient_y
            gradient = np.concatenate((gradient_x, gradient_y), axis=2)

            # We can trust the gradient outside of flow_mask_gradient_img
            # We assume the gradient within flow_mask_gradient_img is 0.
            gradient[flow_mask_gradient_img, :] = 0

            # Complete the flow
            imgSrc_gy = gradient[:, :, 2 : 4]
            imgSrc_gy = imgSrc_gy[0 : imgH - 1, :, :]
            imgSrc_gx = gradient[:, :, 0 : 2]
            imgSrc_gx = imgSrc_gx[:, 0 : imgW - 1, :]
            compFlow[:, :, :, i] = Poisson_blend(flow, imgSrc_gx, imgSrc_gy, flow_mask_img, edge[:, :, i])

        else:
            flow[:, :, 0] = rf.regionfill(flow[:, :, 0], flow_mask_img)
            flow[:, :, 1] = rf.regionfill(flow[:, :, 1], flow_mask_img)
            compFlow[:, :, :, i] = flow

        # Flow visualization.
        flow_img = utils.flow_viz.flow_to_image(compFlow[:, :, :, i])
        flow_img = Image.fromarray(flow_img)

        # Saves the flow and flow_img.
        # flow_img.save(os.path.join('testImg','flow_comp', mode + '_png', '%05d.png'%i))
        # utils.frame_utils.writeFlow(os.path.join('testImg','flow_comp', mode + '_flo', '%05d.flo'%i), compFlow[:, :, :, i])

    return compFlow


def edge_completion(args, EdgeGenerator, corrFlow, flow_mask, mode):
    """Calculate flow edge and complete it.
    """

    if mode not in ['forward', 'backward']:
        raise NotImplementedError

    imgH, imgW, _, nFrame = corrFlow.shape
    Edge = np.empty(((imgH, imgW, 0)), dtype=np.float32)

    for i in range(nFrame):
        print("Completing {0} flow edge {1:2d} <---> {2:2d}".format(mode, i, i + 1), '\r', end='')
        flow_mask_img = flow_mask[:, :, i] if mode == 'forward' else flow_mask[:, :, i + 1]

        flow_img_gray = (corrFlow[:, :, 0, i] ** 2 + corrFlow[:, :, 1, i] ** 2) ** 0.5
        flow_img_gray = flow_img_gray / flow_img_gray.max()

        edge_corr = canny(flow_img_gray, sigma=2, mask=(1 - flow_mask_img).astype(np.bool))
        edge_completed = infer(args, EdgeGenerator, torch.device('cuda:0'), flow_img_gray, edge_corr, flow_mask_img)
        Edge = np.concatenate((Edge, edge_completed[..., None]), axis=-1)

    return Edge
def my_video_completion(args,RAFT_model,video,maskls,outroot,fps,n):
    # Obtains imgH, imgW and nFrame.
    imgH, imgW = np.array(maskls[0]).shape[:2]
    nFrame = len(maskls)

    # Loads video.

    video = torch.stack(video, dim=0)
    video = video.to('cuda')

    # Calcutes the corrupted flow.
    corrFlowF = calculate_flow(args, RAFT_model, video, 'forward')
    corrFlowB = calculate_flow(args, RAFT_model, video, 'backward')
    print('\nFinish flow prediction.')

    # Makes sure video is in BGR (opencv) format.
    video = video.permute(2, 3, 1, 0).cpu().numpy()[:, :, ::-1, :] / 255.
    # Loads masks.
    mask = []
    flow_mask = []
    for mask_img in maskls:
        # Dilate 15 pixel so that all known pixel is trustworthy
        flow_mask_img = scipy.ndimage.binary_dilation(mask_img, iterations=15)
        # Close the small holes inside the foreground objects
        flow_mask_img = cv2.morphologyEx(flow_mask_img.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((21, 21),np.uint8)).astype(np.bool)
        flow_mask_img = scipy.ndimage.binary_fill_holes(flow_mask_img).astype(np.bool)
        flow_mask.append(flow_mask_img)

    # mask indicating the missing region in the video.
    mask = np.stack(maskls, -1).astype(np.bool)
    flow_mask = np.stack(flow_mask, -1).astype(np.bool)
    FlowF_edge, FlowB_edge = None, None

    # Completes the flow.
    videoFlowF = complete_flow(args, corrFlowF, flow_mask, 'forward', FlowF_edge)
    videoFlowB = complete_flow(args, corrFlowB, flow_mask, 'backward', FlowB_edge)
    print('\nFinish flow completion.')

    iter = 0
    mask_tofill = mask
    video_comp = video

    # Image inpainting model.
    deepfill = DeepFillv1(pretrained_model=args.deepfill_model, image_shape=[imgH, imgW])

    # We iteratively complete the video.
    while(np.sum(mask_tofill) > 0):
        # create_dir(os.path.join('testImg', 'frame_comp_' + str(iter)))

        # Color propagation.
        video_comp, mask_tofill, _ = get_flowNN(args,
                                      video_comp,
                                      mask_tofill,
                                      videoFlowF,
                                      videoFlowB,
                                      None,
                                      None)

        for i in range(nFrame):
            mask_tofill[:, :, i] = scipy.ndimage.binary_dilation(mask_tofill[:, :, i], iterations=2)
            img = video_comp[:, :, :, i] * 255
            # Green indicates the regions that are not filled yet.
            img[mask_tofill[:, :, i]] = [0, 255, 0]
            # cv2.imwrite(os.path.join('testImg', 'frame_comp_' + str(iter), '%05d.png'%i), img)

        # video_comp_ = (video_comp * 255).astype(np.uint8).transpose(3, 0, 1, 2)[:, :, :, ::-1]
        #imageio.mimwrite(os.path.join(args.outroot, 'frame_comp_' + str(iter), 'intermediate_{0}.mp4'.format(str(iter))), video_comp_, fps=12, quality=8, macro_block_size=1)
        # imageio.mimsave(os.path.join(args.outroot, 'frame_comp_' + str(iter), 'intermediate_{0}.gif'.format(str(iter))), video_comp_, format='gif', fps=12)
        mask_tofill, video_comp = spatial_inpaint(deepfill, mask_tofill, video_comp)
        iter += 1
    imgls=[]
    create_dir('tmp')
    create_dir(os.path.join('tmp','video'+str(n)))
    video_comp_ = (video_comp * 255).astype(np.uint8).transpose(3, 0, 1, 2)[:, :, :, ::-1]
    for i in range(nFrame):
        img = video_comp[:, :, :, i] * 255
        # imgls.append(img)
        cv2.imwrite(os.path.join('tmp','video'+str(n), '%05d.png'%i), img)
        imageio.mimwrite(outroot, video_comp_, fps=fps, quality=8, macro_block_size=1)
        # imageio.mimsave(os.path.join(args.outroot, 'frame_comp_' + 'final', 'final.gif'), video_comp_, format='gif', fps=12)
    return imgls
def img_completion(args,RAFT_model,video,maskls):
    # Obtains imgH, imgW and nFrame.
    imgH, imgW = np.array(maskls[0]).shape[:2]
    nFrame = len(maskls)

    # Loads video.

    video = torch.stack(video, dim=0)
    video = video.to('cuda')

    # Calcutes the corrupted flow.
    corrFlowF = calculate_flow(args, RAFT_model, video, 'forward')
    corrFlowB = calculate_flow(args, RAFT_model, video, 'backward')
    print('\nFinish flow prediction.')

    # Makes sure video is in BGR (opencv) format.
    video = video.permute(2, 3, 1, 0).cpu().numpy()[:, :, ::-1, :] / 255.
    # Loads masks.
    mask = []
    flow_mask = []
    for mask_img in maskls:
        # Dilate 15 pixel so that all known pixel is trustworthy
        flow_mask_img = scipy.ndimage.binary_dilation(mask_img, iterations=15)
        # Close the small holes inside the foreground objects
        flow_mask_img = cv2.morphologyEx(flow_mask_img.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((21, 21),np.uint8)).astype(np.bool)
        flow_mask_img = scipy.ndimage.binary_fill_holes(flow_mask_img).astype(np.bool)
        flow_mask.append(flow_mask_img)

    # mask indicating the missing region in the video.
    mask = np.stack(maskls, -1).astype(np.bool)
    flow_mask = np.stack(flow_mask, -1).astype(np.bool)
    FlowF_edge, FlowB_edge = None, None

    # Completes the flow.
    videoFlowF = complete_flow(args, corrFlowF, flow_mask, 'forward', FlowF_edge)
    videoFlowB = complete_flow(args, corrFlowB, flow_mask, 'backward', FlowB_edge)
    print('\nFinish flow completion.')

    iter = 0
    mask_tofill = mask
    video_comp = video

    # Image inpainting model.
    deepfill = DeepFillv1(pretrained_model=args.deepfill_model, image_shape=[imgH, imgW])

    # We iteratively complete the video.
    while(np.sum(mask_tofill) > 0):
        # create_dir(os.path.join('testImg', 'frame_comp_' + str(iter)))

        # Color propagation.
        video_comp, mask_tofill, _ = get_flowNN(args,
                                      video_comp,
                                      mask_tofill,
                                      videoFlowF,
                                      videoFlowB,
                                      None,
                                      None)

        for i in range(nFrame):
            mask_tofill[:, :, i] = scipy.ndimage.binary_dilation(mask_tofill[:, :, i], iterations=2)
            img = video_comp[:, :, :, i] * 255
            # Green indicates the regions that are not filled yet.
            img[mask_tofill[:, :, i]] = [0, 255, 0]
            # cv2.imwrite(os.path.join('testImg', 'frame_comp_' + str(iter), '%05d.png'%i), img)

        # video_comp_ = (video_comp * 255).astype(np.uint8).transpose(3, 0, 1, 2)[:, :, :, ::-1]
        #imageio.mimwrite(os.path.join(args.outroot, 'frame_comp_' + str(iter), 'intermediate_{0}.mp4'.format(str(iter))), video_comp_, fps=12, quality=8, macro_block_size=1)
        # imageio.mimsave(os.path.join(args.outroot, 'frame_comp_' + str(iter), 'intermediate_{0}.gif'.format(str(iter))), video_comp_, format='gif', fps=12)
        mask_tofill, video_comp = spatial_inpaint(deepfill, mask_tofill, video_comp)
        iter += 1
    create_dir('tmp')
    for i in range(nFrame):
        img = video_comp[:, :, :, i] * 255
        cv2.imwrite('tmp/preview.png', img)
        # imageio.mimwrite(outroot, video_comp_, fps=fps, quality=8, macro_block_size=1)
        # imageio.mimsave(os.path.join(args.outroot, 'frame_comp_' + 'final', 'final.gif'), video_comp_, format='gif', fps=12)