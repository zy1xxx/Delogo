from builtins import enumerate, exit
import sys
sys.path.insert(1,"./FGVC/tool")
from torch.multiprocessing import Pool, set_start_method
from multiprocessing import Process
from my_completion import my_video_completion,initialize_RAFT
import cv2
import argparse
import numpy as np
import torch
sys.path.append("RIFE")
# from myREFI2 import run_RIFE
from my_RIFE import run_RIFE
import os
from getShot import slice_video
try:
    set_start_method('spawn')
except RuntimeError:
    pass
import math
import ast
class VideoReadError(Exception):
    pass
dim=(160,160)

resizeFlag=False

args=argparse.Namespace(H_scale=2, Nonlocal=False, W_scale=2, alpha=0.1, alternate_corr=False, consistencyThres=np.inf, deepfill_model='./FGVC/weight/imagenet_deepfill.pth', edge_completion_model='./FGVC/weight/edge_completion.pth', edge_guide=False, mixed_precision=False, mode='object_removal', model='./FGVC/weight/raft-things.pth', seamless=False, small=False)

modelDir='./RIFE/train_log'

def getMask(maskimg):
    for i in range(heightMask):
        for j in range(widthMask):
            maskimg[ymask+i][xmask+j]=255
    # cv2.imwrite("./masktest.png",maskimg)
    return maskimg
def initGlobalVeriable():
    global heightMask,widthMask,ymask,xmask,widthCut,heightCut,xcut,ycut,cap,shape,fps,RAFT_model,mask,maskimg,modelP,RIFEmodel
    #init mask coord and rec 
    heightMask=userArgs.maskRec[1]
    widthMask=userArgs.maskRec[0]
    ymask=userArgs.maskcoords[1]
    xmask=userArgs.maskcoords[0]
    #init cut
    if not userArgs.cutCoords:
        if userArgs.captionMode:
            widthCut=math.ceil(widthMask/8*1.5)*8
            heightCut=widthCut
            xCutTmp=xmask-int(widthCut/2-widthMask/2)
            xcut=xCutTmp if xCutTmp>=0 else 0
            yCutTmp=ymask-int(heightCut/2)
            ycut=yCutTmp if yCutTmp>=0 else 0
        else:
            heightTmp=math.ceil(heightMask/8)*8 
            widthTmp=math.ceil(widthMask/8)*8
            widthCut=max(heightTmp*2,widthTmp*2)
            heightCut=widthCut
            xcut=0
            ycut=0
    else:
        heightCut=math.ceil(userArgs.cutRec[1]/8)*8
        widthCut=math.ceil(userArgs.cutRec[0]/8)*8
        xcut=userArgs.cutCoords[0]
        ycut=userArgs.cutCoords[1]
    print("coords",xcut,ycut,widthCut,heightCut)
    cap = cv2.VideoCapture(userArgs.path)
    if not cap.isOpened():
        raise VideoReadError(f'视频文件读取失败，请检查 {userArgs.path}')
    ret,img=cap.read()
    shape=img.shape   
    fps=cap.get(cv2.CAP_PROP_FPS)
    #init model
    RAFT_model=initialize_RAFT(args)
    if userArgs.extractFlag:
        RIFEmodel=iniRIFEmodel()
    #init mask frame
    maskimg=np.zeros((shape[0],shape[1]),dtype=np.uint8)
    mask=getMask(maskimg)


def getSplitLs(shotls):
    finalLs=[]
    for index,para in enumerate(shotls):
        count=int((para[1]-para[0]))
        if count<30:
            finalLs.append(para)
            continue
        ctnPiece=int(count/30)
        piece=int(count/ctnPiece)
        start=para[0]
        end=para[1]
        ctn=start
        for i in range(ctnPiece):
            tmp=ctn+piece
            if tmp>end:
                finalLs.append((ctn,end))
            else:
                finalLs.append((ctn,tmp))
            ctn=tmp+1
    return finalLs
# @profile
def getcompletion(finalLs):
    p = Pool(8)
    startFrame=finalLs[0][0]
    cap.set(1,startFrame)
    for index,para in enumerate(finalLs):
        startFrame=para[0]
        endFrame=para[1]
        ctn=0
        maskls=[]
        video=[]
        while ctn<=endFrame-startFrame:
            ret,frame=cap.read()
            if not ret:
                print("can't get origin frame")
                continue
            if userArgs.extractFlag:
                if ctn%2!=0:
                    if ctn==endFrame-startFrame:
                        resized1=frame[ycut:ycut+heightCut,xcut:xcut+widthCut]
                        resized1=getResize(resized1)
                        resized1RGB=cv2.cvtColor(resized1,cv2.COLOR_BGR2RGB)
                        video.append(torch.from_numpy(resized1RGB.astype(np.uint8)).permute(2, 0, 1).float())
                        resized2=mask[ycut:ycut+heightCut,xcut:xcut+widthCut]
                        resized2=getResize(resized2)
                        maskls.append(resized2)
                    ctn+=1
                    continue
            resized1=frame[ycut:ycut+heightCut,xcut:xcut+widthCut]
            resized1=getResize(resized1)
            resized1RGB=cv2.cvtColor(resized1,cv2.COLOR_BGR2RGB)
            video.append(torch.from_numpy(resized1RGB.astype(np.uint8)).permute(2, 0, 1).float())
            resized2=mask[ycut:ycut+heightCut,xcut:xcut+widthCut]
            resized2=getResize(resized2)
            maskls.append(resized2)
            ctn+=1
        print("get started")
        outroot=os.path.join('tmp','video'+str(index)+'.mp4')
        p.apply_async(my_video_completion,args=(args,RAFT_model,video,maskls,outroot,fps,index))
    p.close()
    p.join()

def getResize(img):
    if resizeFlag:
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        return resized
    return img
def checkIfResize():
    global resizeFlag
    if heightCut>dim[1] or widthCut>dim[0]:
        resizeFlag=True
    else:
        resizeFlag=False
def resizeBack(img):
    resized = cv2.resize(img, (widthCut,heightCut), interpolation = cv2.INTER_AREA)
    return resized
def getSplitLs(shotls):
    finalLs=[]
    for para in shotls:
        count=int((para[1]-para[0]))
        if count<30:
            finalLs.append(para)
            continue
        ctnPiece=int(round(count/30))
        piece=int(count/ctnPiece)
        start=para[0]
        end=para[1]
        ctn=start
        for i in range(ctnPiece):
            tmp=ctn+piece
            if tmp>end:
                finalLs.append((ctn,end))
            else:
                finalLs.append((ctn,tmp))
            ctn=tmp+1
    return finalLs
def preProcess(shotls):
    finalLs=[]
    i=0
    finalLsctn=0
    while i<len(shotls):
        if shotls[i][0]+5>shotls[i][1]:
            if i+1<len(shotls):
                finalLs.append((shotls[i][0],shotls[i+1][1]))
            else:
                finalLs.append(shotls[i])
                return finalLs
            i+=1
        else:
            finalLs.append(shotls[i])
        i+=1
        finalLsctn+=1
    return finalLs
def iniRIFEmodel():
    try:
        try:
            from model.RIFE_HDv2 import Model
            model = Model()
            model.load_model(modelDir, -1)
            print("Loaded v2.x HD model.")
        except:
            from train_log.RIFE_HDv3 import Model
            model = Model()
            model.load_model(modelDir, -1)
            print("Loaded v3.x HD model.")
    except:
        from model.RIFE_HD import Model
        model = Model()
        model.load_model(modelDir, -1)
        print("Loaded v1.x HD model")
    model.eval()
    model.device()
    return model
def getRIFE(finalLs):
    p = Pool(8)
    for index,para in enumerate(finalLs):
        video='tmp/video'+str(index)+'.mp4'
        p.apply_async(run_RIFE,args=(RIFEmodel,video,index))
    print("start RIFE")
    p.close()
    p.join()
    print("RIFE end")
def getOverlayVideo(finalLs):
    if userArgs.extractFlag:
        fullvideoPath="./tmp/videoRIFE"
    else:
        fullvideoPath="./tmp/video"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(userArgs.output,fourcc, fps, (shape[1],shape[0]),True)
    for i,shot in enumerate(finalLs):
        cap2=cv2.VideoCapture(fullvideoPath+str(i)+".mp4")
        print(fullvideoPath+str(i)+".mp4")
        startframeCtn=int(shot[0])
        endframeCtn=int(shot[1])
        cap.set(1,startframeCtn)
        for frameCtn in range(startframeCtn,endframeCtn+1):
            ret,frame=cap.read()
            ret1,resized=cap2.read()    
            if (not ret) or (not ret1):
                print("can't get frame")
                continue
            resizeback=resizeBack(resized)
            if not userArgs.captionMode:
                offset=50
                if offset<heightMask:
                    frame[ycut:ycut+heightMask+offset,xcut:xcut+widthMask+offset,:]=resizeback[0:heightMask+offset,0:widthMask+offset,:]
                else:
                    frame[ycut:ycut+heightCut,xcut:xcut+widthCut,:]=resizeback[0:heightCut,0:widthCut,:]
            else:
                frame[ycut:ycut+heightCut,xcut:xcut+widthCut,:]=resizeback[0:heightCut,0:widthCut,:]
            # cv2.imwrite("tmp.png",frame)
            out.write(frame)
# @profile
def run(path='lena2.mp4',maskcoords=(0,0),maskRec=(160,160),output='lena_out.mp4',frameInterval=(0,15),extractFlag=False,captionMode=False,patchSplit=True, \
        cutCoords=None,cutRec=None):    
    """ a tool to remove logo or caption in a video
        
        input a video and its logo's coords,width,height ,then output a logo removed video
        input a video and its caption's coords,width,height ,then output a caption removed video

    Args:
        path: input video path
        maskcoords: mask coords,format:(x,y)
        maskRec: mask width and height,format:(width,height)
        output: output video path
        frameInterval: the start and end frame ctn,format:(start,end),(0,0) means process the whole video
        extractFlag: if extract frames (every two frames extract one frame).True will speed up the progress
        captionMode: if the removed item is caption
        patchSplit: if the shot is cut into about 30 frames each piece
        cutCoords: the cut coords,format:(x,y),None means the code will help you to decide the coords and width and height
        cutRec: the cut width and height,format:(width,height)


    Returns:
        output path
    """
    global userArgs
    userArgs=argparse.Namespace(path='lena2.mp4',maskcoords=(0,0),maskRec=(160,160),output='lena_out.mp4',frameInterval=(0,15),extractFlag=False,captionMode=False)
    userArgs.path=path
    userArgs.maskcoords=maskcoords
    userArgs.maskRec=maskRec
    userArgs.output=output
    userArgs.frameInterval=frameInterval
    userArgs.extractFlag=extractFlag
    userArgs.captionMode=captionMode
    userArgs.cutCoords=cutCoords
    userArgs.cutRec=cutRec
    print(userArgs)
    initGlobalVeriable()
    checkIfResize()
    if userArgs.captionMode:
        finalLs=[userArgs.frameInterval]
    else:
        print("start shot")
        if userArgs.frameInterval[1]==0:
            shotls=slice_video(userArgs.path,0,cap.get(7)-1)
        else:
            shotls=slice_video(userArgs.path,userArgs.frameInterval[0],userArgs.frameInterval[1])#get shot
        print(shotls)
        shotls=preProcess(shotls)
        if patchSplit:
            finalLs=getSplitLs(shotls)
        else:
            finalLs=shotls 
        print(finalLs)
    getcompletion(finalLs)#use the AI model to remove the logo and caption
    if userArgs.extractFlag:
        getRIFE(finalLs)#recovery the removed frames with a AI model
    getOverlayVideo(finalLs)#put the cut video on the origin video with right coords
    os.system('rm -r tmp')
    print("video has been outputed to :",userArgs.output)
    return userArgs.output
if __name__ == '__main__':
    #examples
    # run(path='test3.mp4',extractFlag=True,frameInterval=(0,300),maskcoords=(16,26),maskRec=(326-16,160-26),output="test3_output.mp4") # test3 图标
    # run(captionMode=True,extractFlag=False,frameInterval=(0,18),maskcoords=(145,429),maskRec=(338-145,491-429),cutCoords=(109,396),cutRec=(max(383-109,579-396),max(383-109,579-396))) #lena2 字幕
    # run(path="https://mira-1255830993.cos.ap-shanghai.myqcloud.com/public2/lena2.mp4",captionMode=True,extractFlag=False,frameInterval=(0,0),maskcoords=(0,0),maskRec=(104,148),output="lena2_out.mp4") #lena2 图标       
    # run(path='test1.mp4',captionMode=False,extractFlag=True,frameInterval=(0,0),maskcoords=(41,37),maskRec=(155-41,158-37),output='test132_out.mp4')  #test1
    # run(path='test2.mp4',captionMode=False,patchSplit=False,extractFlag=False,frameInterval=(0,0),maskcoords=(0,0),maskRec=(209,240),output="test2_out.mp4")  #test1
    # run(path='../test8.mp4',captionMode=False,patchSplit=False,extractFlag=False,frameInterval=(0,250),maskcoords=(41,47),maskRec=(261-41,267-47),output="test88_out.mp4")  #test5
    # run(path='../test8.mp4',captionMode=False,patchSplit=False,extractFlag=False,frameInterval=(0,250),maskcoords=(0,0),maskRec=(261,267),output="test88_out.mp4")  #test5
    # run(path='../test6.mp4',captionMode=False,patchSplit=False,extractFlag=False,frameInterval=(0,0),maskcoords=(0,0),maskRec=(136,130),output="test6_out.mp4")  
    # run(path='../test7.mp4',captionMode=False,patchSplit=False,extractFlag=True,frameInterval=(0,0),maskcoords=(0,0),maskRec=(144,132),output="test77777_out.mp4")  
    #run(path='../test9.mp4',captionMode=False,patchSplit=False,extractFlag=False,frameInterval=(0,250),maskcoords=(0,0),maskRec=(184,180),output="test9_out.mp4")  
    # run(path='../test10.mp4',captionMode=False,patchSplit=True,extractFlag=True,frameInterval=(0,0),maskcoords=(0,0),maskRec=(170,185),output="test10_out.mp4") 

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='lena2.mp4', help='the origin video path')
    parser.add_argument('--maskcoords', default="(0,0)",help='the mask coords,format "(x,y)"')
    parser.add_argument('--maskRec', default="(160,160)",help='the mask width and height,format "(width,height)"')
    parser.add_argument('--output', default='lena_out.mp4',help='the output video path')
    parser.add_argument('--frameInterval', default="(0,0)",help='the frame start and end,format "(start,end)"')
    parser.add_argument('--extractFlag',action="store_true",help='the flag if extract frames')
    parser.add_argument('--captionMode',action="store_true",help="the flag if there's only one shot")
    parser.add_argument('--patchSplit',action="store_true",help="if split into patch")
    parser.add_argument('--cutCoords',default="None",help='the Cut coords,format "(x,y)". if None the code will decide for you')
    parser.add_argument('--cutRec',default="None",help='the Cut width and height,format "(width,height)" .if None the code will decide for you')
    cmdArgs=parser.parse_args()

    if cmdArgs.cutCoords=="None":
        tmpCutCoords=None
    else:
        tmpCutCoords=ast.literal_eval(cmdArgs.cutCoords)
        if tmpCutCoords[0]<0 or tmpCutCoords[1]<0:
            print("the cut coords must >= 0")
        raise ValueError
    if cmdArgs.cutRec=="None":
        tmpCutRec=None
    else:
        tmpCutRec=ast.literal_eval(cmdArgs.cutRec)
        if tmpCutRec[0]<0 or tmpCutRec[1]<0:
            print("the cut width and height must >= 0")
        raise ValueError
    if not os.path.exists(cmdArgs.path):
        print("the input path file doesn't exist,please check your path")
        raise FileNotFoundError
    tmpMaskCoords=ast.literal_eval(cmdArgs.maskcoords)
    if tmpMaskCoords[0]<0 or tmpMaskCoords[1]<0:
        print("the mask coords must >= 0")
        raise ValueError
    tmpMaskRec=ast.literal_eval(cmdArgs.maskRec)
    if tmpMaskRec[0]<0 or tmpMaskRec[1]<0:
        print("the mask width and height must >= 0")
        raise ValueError
    tmpframeInterval=ast.literal_eval(cmdArgs.frameInterval)
    if tmpframeInterval[0]<0:
        print("the start frame cnt must >= 0")
        raise ValueError
    
    run(path=cmdArgs.path,maskcoords=tmpMaskCoords,maskRec=tmpMaskRec,output=cmdArgs.output,frameInterval=tmpframeInterval,
        extractFlag=cmdArgs.extractFlag,captionMode=cmdArgs.captionMode,patchSplit=cmdArgs.patchSplit,cutCoords=tmpCutCoords,cutRec=tmpCutRec
    )
    exit(0)

    


    






