import os
import cv2
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
import warnings
import _thread
import skvideo.io
from queue import Queue, Empty
from model.pytorch_msssim import ssim_matlab
import argparse
warnings.filterwarnings("ignore")

def transferAudio(sourceVideo, targetVideo):
    import shutil
    import moviepy.editor
    tempAudioFileName = "./temp/audio.mkv"

    # split audio from original video file and store in "temp" directory
    if True:
        # clear old "temp" directory if it exits
        if os.path.isdir("temp"):
            # remove temp directory
            shutil.rmtree("temp")
        # create new "temp" directory
        os.makedirs("temp")
        # extract audio from video
        os.system('ffmpeg -y -i "{}" -c:a copy -vn {}'.format(sourceVideo, tempAudioFileName))

    targetNoAudio = os.path.splitext(targetVideo)[0] + "_noaudio" + os.path.splitext(targetVideo)[1]
    os.rename(targetVideo, targetNoAudio)
    # combine audio file and new video file
    os.system('ffmpeg -y -i "{}" -i {} -c copy "{}"'.format(targetNoAudio, tempAudioFileName, targetVideo))

    if os.path.getsize(targetVideo) == 0: # if ffmpeg failed to merge the video and audio together try converting the audio to aac
        tempAudioFileName = "./temp/audio.m4a"
        os.system('ffmpeg -y -i "{}" -c:a aac -b:a 160k -vn {}'.format(sourceVideo, tempAudioFileName))
        os.system('ffmpeg -y -i "{}" -i {} -c copy "{}"'.format(targetNoAudio, tempAudioFileName, targetVideo))
        if (os.path.getsize(targetVideo) == 0): # if aac is not supported by selected format
            os.rename(targetNoAudio, targetVideo)
            print("Audio transfer failed. Interpolated video will have no audio")
        else:
            print("Lossless audio transfer failed. Audio was transcoded to AAC (M4A) instead.")

            # remove audio-less video
            os.remove(targetNoAudio)
    else:
        os.remove(targetNoAudio)

    # remove temp directory
    shutil.rmtree("temp")

def clear_write_buffer(user_args, write_buffer):
    cnt = 0
    while True:
        item = write_buffer.get()
        if item is None:
            break
        if user_args.png:
            cv2.imwrite('vid_out/{:0>7d}.png'.format(cnt), item[:, :, ::-1])
            cnt += 1
        else:
            vid_out.write(item[:, :, ::-1])

def build_read_buffer(user_args, read_buffer, videogen):
    try:
        for frame in videogen:
             if not user_args.img is None:
                  frame = cv2.imread(os.path.join(user_args.img, frame))[:, :, ::-1].copy()
             if user_args.montage:
                  frame = frame[:, left: left + w]
             read_buffer.put(frame)
    except:
        pass
    read_buffer.put(None)

def make_inference(I0, I1, n):
    global model
    middle = model.inference(I0, I1, args.scale)
    if n == 1:
        return [middle]
    first_half = make_inference(I0, middle, n=n//2)
    second_half = make_inference(middle, I1, n=n//2)
    if n%2:
        return [*first_half, middle, *second_half]
    else:
        return [*first_half, *second_half]

def pad_image(img):
    if(args.fp16):
        return F.pad(img, padding).half()
    else:
        return F.pad(img, padding)

args=argparse.Namespace(UHD=False, exp=1, ext='mp4', fp16=False, fps=None, img=None, modelDir='./RIFE/train_log', montage=False, output='./tmp/RIFEvideo.mp4', png=False, scale=1.0, skip=False, video='./tmp/fullvideo.mp4')
def run_RIFE(usermode,video,index):
    global padding
    global vid_out
    global model

    model=usermode

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
    

    videoCapture = cv2.VideoCapture(video)
    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    tot_frame = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
    videoCapture.release()
    if args.fps is None:
        fpsNotAssigned = True
        args.fps = fps * (2 ** args.exp)
    else:
        fpsNotAssigned = False
    videogen = skvideo.io.vreader(video)
    lastframe = next(videogen)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video_path_wo_ext, ext = os.path.splitext(video)
    print('{}.{}, {} frames in total, {}FPS to {}FPS'.format(video_path_wo_ext, args.ext, tot_frame, fps, args.fps))

    h, w, _ = lastframe.shape
    vid_out_name = 'tmp/'+"videoRIFE"+str(index)+'.mp4'
    
    vid_out = cv2.VideoWriter(vid_out_name, fourcc, args.fps, (w, h))


    tmp = max(32, int(32 / args.scale))
    ph = ((h - 1) // tmp + 1) * tmp
    pw = ((w - 1) // tmp + 1) * tmp
    padding = (0, pw - w, 0, ph - h)
    pbar = tqdm(total=tot_frame)
    skip_frame = 1

    write_buffer = Queue(maxsize=500)
    read_buffer = Queue(maxsize=500)
    _thread.start_new_thread(build_read_buffer, (args, read_buffer, videogen))
    _thread.start_new_thread(clear_write_buffer, (args, write_buffer))

    I1 = torch.from_numpy(np.transpose(lastframe, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
    I1 = pad_image(I1)

    while True:
        frame = read_buffer.get()
        if frame is None:
            break
        I0 = I1
        I1 = torch.from_numpy(np.transpose(frame, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
        I1 = pad_image(I1)
        I0_small = F.interpolate(I0, (32, 32), mode='bilinear', align_corners=False)
        I1_small = F.interpolate(I1, (32, 32), mode='bilinear', align_corners=False)
        ssim = ssim_matlab(I0_small[:, :3], I1_small[:, :3])

        if ssim > 0.995:
            if skip_frame % 100 == 0:
                print("\nWarning: Your video has {} static frames, skipping them may change the duration of the generated video.".format(skip_frame))
            skip_frame += 1
            if args.skip:
                pbar.update(1)
                continue

        if ssim < 0.2:
            output = []
            for i in range((2 ** args.exp) - 1):
                output.append(I0)
            '''
            output = []
            step = 1 / (2 ** args.exp)
            alpha = 0
            for i in range((2 ** args.exp) - 1):
                alpha += step
                beta = 1-alpha
                output.append(torch.from_numpy(np.transpose((cv2.addWeighted(frame[:, :, ::-1], alpha, lastframe[:, :, ::-1], beta, 0)[:, :, ::-1].copy()), (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.)
            '''
        else:
            output = make_inference(I0, I1, 2**args.exp-1) if args.exp else []

        write_buffer.put(lastframe)
        for mid in output:
            mid = (((mid[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0)))
            write_buffer.put(mid[:h, :w])
        
        pbar.update(1)
        lastframe = frame

    write_buffer.put(lastframe)
    import time
    while(not write_buffer.empty()):
        time.sleep(0.1)
    pbar.close()
    if not vid_out is None:
        vid_out.release()
    # move audio to new video file if appropriate
    # if args.png == False and fpsNotAssigned == True and not args.skip and not args.video is None:
    #     try:
    #         transferAudio(args.video, vid_out_name)
    #     except:
    #         print("Audio transfer failed. Interpolated video will have no audio")
    #         targetNoAudio = os.path.splitext(vid_out_name)[0] + "_noaudio" + os.path.splitext(vid_out_name)[1]
    #         os.rename(targetNoAudio, vid_out_name)
    del model
