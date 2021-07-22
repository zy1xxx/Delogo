from builtins import enumerate, exit
import time
import sys
from imageio.core.functions import imread
sys.path.insert(1,"./FGVC/tool")
from torch.multiprocessing import Pool, set_start_method
from multiprocessing import Process
from my_completion import my_video_completion,initialize_RAFT,img_completion
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
class Delogo(object):
    FGVC_args=argparse.Namespace(H_scale=2, Nonlocal=False, W_scale=2, alpha=0.1, alternate_corr=False, consistencyThres=np.inf, deepfill_model='./FGVC/weight/imagenet_deepfill.pth', edge_completion_model='./FGVC/weight/edge_completion.pth', edge_guide=False, mixed_precision=False, mode='object_removal', model='./FGVC/weight/raft-things.pth', seamless=False, small=False)
    RIFEmodelDir='./RIFE/train_log'
    dim=(160,160)
    def init_RIFE_model(self):
        try:
            try:
                from model.RIFE_HDv2 import Model
                model = Model()
                model.load_model(self.RIFEmodelDir, -1)
                print("Loaded v2.x HD model.")
            except:
                from train_log.RIFE_HDv3 import Model
                model = Model()
                model.load_model(self.RIFEmodelDir, -1)
                print("Loaded v3.x HD model.")
        except:
            from model.RIFE_HD import Model
            model = Model()
            model.load_model(self.RIFEmodelDir, -1)
            print("Loaded v1.x HD model")
        model.eval()
        model.device()
        return model
    def init_FGVC_model(self):
        return initialize_RAFT(self.FGVC_args)
    def initVeriables(self,path,output,maskcoords=(0,0),maskRec=(160,160),cutCoords=None,cutRec=None):
        #init mask
        self.heightMask=maskRec[1]
        self.widthMask=maskRec[0]
        self.ymask=maskcoords[1]
        self.xmask=maskcoords[0]
        #init cut
        if not cutCoords:
            if self.captionMode:
                self.widthCut=math.ceil(self.widthMask/8*1.5)*8
                self.heightCut=self.widthCut
                self.xCutTmp=self.xmask-int(self.widthCut/2-self.widthMask/2)
                self.xcut=self.xCutTmp if self.xCutTmp>=0 else 0
                self.yCutTmp=self.ymask-int(self.heightCut/2)
                self.ycut=self.yCutTmp if self.yCutTmp>=0 else 0
            else:
                self.heightTmp=math.ceil(self.heightMask/8)*8 
                self.widthTmp=math.ceil(self.widthMask/8)*8
                self.widthCut=max(self.heightTmp*2,self.widthTmp*2)
                self.heightCut=self.widthCut
                self.xcut=0 if self.widthCut>self.xmask+self.widthMask else self.xmask
                self.ycut=0 if self.heightMask>self.ymask+self.heightMask else self.ymask
                
        else:
            self.heightCut=math.ceil(cutRec[1]/8)*8
            self.widthCut=math.ceil(cutRec[0]/8)*8
            self.xcut=cutCoords[0]
            self.ycut=cutCoords[1]
        print("cut coords",self.xcut,self.ycut,self.widthCut,self.heightCut)
        #init file path
        self.path=path
        self.output=output
        #init video info
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise VideoReadError(f'视频文件读取失败，请检查 {self.path}')
        ret,img=self.cap.read()
        self.shape=img.shape   
        print(self.shape[0],self.shape[1])
        self.fps=self.cap.get(cv2.CAP_PROP_FPS)

        #check mask and cut if vaild
        if self.widthCut+self.xcut>self.shape[1] or self.heightCut+self.ycut>self.shape[0]:
            print("the cut is out of bound")
            raise ValueError
        elif self.widthMask+self.xmask>self.shape[1] or self.heightMask+self.ymask>self.shape[0]:
            print("the mask is out of bound")
            raise ValueError
        
        #init mask frame
        maskimg=np.zeros((self.shape[0],self.shape[1]),dtype=np.uint8)
        self.mask=self.__getMask(maskimg)
    def __getResize(self,img):
        if self.resizeFlag:
            resized = cv2.resize(img, self.dim, interpolation = cv2.INTER_AREA)
            return resized
        return img
    def __getMask(self,maskimg):
        for i in range(self.heightMask):
            for j in range(self.widthMask):
                maskimg[self.ymask+i][self.xmask+j]=255
        # cv2.imwrite("./masktest.png",maskimg)
        return maskimg
    def __init__(self):
        self.FGVCmodel=self.init_FGVC_model()
        self.RIFEmodel=self.init_RIFE_model()
    def __checkIfResize(self):
        if self.heightCut>self.dim[1] or self.widthCut>self.dim[0]:
            self.resizeFlag=True
        else:
            self.resizeFlag=False
    def __preProcess(self,shotls):
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
    def __getSplitLs(self,shotls):
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
    def __getcompletion(self,finalLs):
        p = Pool(8)
        startFrame=finalLs[0][0]
        self.cap.set(1,startFrame)
        for index,para in enumerate(finalLs):
            startFrame=para[0]
            endFrame=para[1]
            ctn=0
            maskls=[]
            video=[]
            while ctn<=endFrame-startFrame:
                ret,frame=self.cap.read()
                if not ret:
                    print("can't get origin frame")
                    continue
                if self.extractFlag:
                    if ctn%2!=0:
                        if ctn==endFrame-startFrame:
                            resized1=frame[self.ycut:self.ycut+self.heightCut,self.xcut:self.xcut+self.widthCut]
                            resized1=self.__getResize(resized1)
                            resized1RGB=cv2.cvtColor(resized1,cv2.COLOR_BGR2RGB)
                            video.append(torch.from_numpy(resized1RGB.astype(np.uint8)).permute(2, 0, 1).float())
                            resized2=self.mask[self.ycut:self.ycut+self.heightCut,self.xcut:self.xcut+self.widthCut]
                            resized2=self.__getResize(resized2)
                            maskls.append(resized2)
                        ctn+=1
                        continue
                resized1=frame[self.ycut:self.ycut+self.heightCut,self.xcut:self.xcut+self.widthCut]
                resized1=self.__getResize(resized1)
                resized1RGB=cv2.cvtColor(resized1,cv2.COLOR_BGR2RGB)
                video.append(torch.from_numpy(resized1RGB.astype(np.uint8)).permute(2, 0, 1).float())
                resized2=self.mask[self.ycut:self.ycut+self.heightCut,self.xcut:self.xcut+self.widthCut]
                resized2=self.__getResize(resized2)
                maskls.append(resized2)
                ctn+=1
            print("create FGVA process: "+str(index))
            outroot=os.path.join('tmp','video'+str(index)+'.mp4')
            p.apply_async(my_video_completion,args=(self.FGVC_args,self.FGVCmodel,video,maskls,outroot,self.fps,index))
        p.close()
        p.join()
    def __getRIFE(self,finalLs):
        p = Pool(8)
        for index in range(len(finalLs)):
            video='tmp/video'+str(index)+'.mp4'
            p.apply_async(run_RIFE,args=(self.RIFEmodel,video,index))
        print("start RIFE")
        p.close()
        p.join()
        print("RIFE end")
    def __getOverlayVideo(self,finalLs):
        if self.extractFlag:
            fullvideoPath="./tmp/videoRIFE"
        else:
            fullvideoPath="./tmp/video"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output,fourcc, self.fps, (self.shape[1],self.shape[0]),True)
        for i,shot in enumerate(finalLs):
            cap2=cv2.VideoCapture(fullvideoPath+str(i)+".mp4")
            print(fullvideoPath+str(i)+".mp4")
            startframeCtn=int(shot[0])
            endframeCtn=int(shot[1])
            self.cap.set(1,startframeCtn)
            for frameCtn in range(startframeCtn,endframeCtn+1):
                ret,frame=self.cap.read()
                ret1,resized=cap2.read()    
                if (not ret) or (not ret1):
                    print("can't get frame")
                    continue
                resizeback=self.__resizeBack(resized)
                if not self.captionMode:
                    offset=50
                    if offset<self.heightMask:
                        frame[self.ycut:self.ycut+self.heightMask+offset,self.xcut:self.xcut+self.widthMask+offset,:]=resizeback[0:self.heightMask+offset,0:self.widthMask+offset,:]
                    else:
                        frame[self.ycut:self.ycut+self.heightCut,self.xcut:self.xcut+self.widthCut,:]=resizeback[0:self.heightCut,0:self.widthCut,:]
                else:
                    frame[self.ycut:self.ycut+self.heightCut,self.xcut:self.xcut+self.widthCut,:]=resizeback[0:self.heightCut,0:self.widthCut,:]
                # cv2.imwrite("tmp.png",frame)
                out.write(frame)
    def run(self,path,output,maskcoords=(0,0),maskRec=(160,160),frameInterval=(0,0),extractFlag=False,captionMode=False,cutCoords=None,cutRec=None,patchSplit=True):
        self.extractFlag=extractFlag
        self.captionMode=captionMode

        self.initVeriables(path,output,maskcoords,maskRec,cutCoords,cutRec)
        self.__checkIfResize()
        if captionMode:
            finalLs=[frameInterval]
        else:
            print("start shot")
            if frameInterval[1]==0:
                shotls=slice_video(path,0,self.cap.get(7)-1)
            else:
                shotls=slice_video(path,frameInterval[0],frameInterval[1])#get shot
            print(shotls)
            shotls=self.__preProcess(shotls)
            if patchSplit:
                finalLs=self.__getSplitLs(shotls)
            else:
                finalLs=shotls 
            print(finalLs)
        self.__getcompletion(finalLs)#use the AI model to remove the logo and caption
        if extractFlag:
            self.__getRIFE(finalLs)#recovery the removed frames with a AI model
        self.__getOverlayVideo(finalLs)#put the cut video on the origin video with right coords
        os.system('rm -r tmp')
        print("video has been outputed to :",output)
        return output
    def __resizeBack(self,img):
        if self.resizeFlag:
            resized = cv2.resize(img, (self.widthCut,self.heightCut), interpolation = cv2.INTER_AREA)
            return resized
        else:
            return img
    
    def preview(self,path,output,maskcoords=(0,0),frameInterval=(0,0),maskRec=(160,160),cutCoords=None,cutRec=None,captionMode=False):
        self.captionMode=captionMode
        self.initVeriables(path,output,maskcoords,maskRec,cutCoords,cutRec)
        self.__checkIfResize()
        #FGVC
        video=[]
        maskls=[]
        self.cap.set(1,frameInterval[0])
        ret,frame=self.cap.read()
        resized1=frame[self.ycut:self.ycut+self.heightCut,self.xcut:self.xcut+self.widthCut]
        resized1=self.__getResize(resized1)
        resized1RGB=cv2.cvtColor(resized1,cv2.COLOR_BGR2RGB)
        video.append(torch.from_numpy(resized1RGB.astype(np.uint8)).permute(2, 0, 1).float())
        resized2=self.mask[self.ycut:self.ycut+self.heightCut,self.xcut:self.xcut+self.widthCut]
        resized2=self.__getResize(resized2)
        maskls.append(resized2)
        img_completion(self.FGVC_args,self.FGVCmodel,video,maskls)
        #overlay
        FGVC_frame=cv2.imread('tmp/preview.png')
        resizeback=self.__resizeBack(FGVC_frame)
        frame[self.ycut:self.ycut+self.heightCut,self.xcut:self.xcut+self.widthCut,:]=resizeback[0:self.heightCut,0:self.widthCut,:]
        cv2.imwrite(output,frame)
        os.system('rm -r tmp')
        print("img has been outputed to :",output)
        return output
if __name__ == '__main__':

    delogo=Delogo()  
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
    tmpMaskCoords=ast.literal_eval(cmdArgs.maskcoords)
    if tmpMaskCoords[0]<0 or tmpMaskCoords[1]<0:
        print("the mask coords must >= 0")
        raise ValueError
    tmpMaskRec=ast.literal_eval(cmdArgs.maskRec)
    if tmpMaskRec[0]<0 or tmpMaskRec[1]<0:
        print("the mask width and height must >= 0")
        raise ValueError
    tmpframeInterval=ast.literal_eval(cmdArgs.frameInterval)
    if tmpframeInterval[0]<0 or tmpframeInterval[1]<0:
        print("the start or end frame cnt must >= 0")
        raise ValueError
    delogo.run(path=cmdArgs.path,maskcoords=tmpMaskCoords,maskRec=tmpMaskRec,output=cmdArgs.output,frameInterval=tmpframeInterval,
        extractFlag=cmdArgs.extractFlag,captionMode=cmdArgs.captionMode,patchSplit=cmdArgs.patchSplit,cutCoords=tmpCutCoords,cutRec=tmpCutRec
    )

    # function example
    # delogo.run(path="lena2.mp4",captionMode=True,extractFlag=False,frameInterval=(0,18),maskcoords=(138,424),maskRec=(337-138,493-424),output="lena2_out_cap.mp4") 
    # delogo.preview(path="lena2.mp4",frameInterval=(0,0),maskcoords=(138,424),maskRec=(337-138,493-424),output="lena2_preview.png",captionMode=True)
    # delogo.preview(path="lena2.mp4",frameInterval=(30,0),maskcoords=(0,0),maskRec=(104,148),output="lena2_preview1.png")
    # delogo.preview(path="lena2.mp4",frameInterval=(60,0),maskcoords=(0,0),maskRec=(104,148),output="lena2_preview2.png")
    # delogo.preview(path="lena2.mp4",frameInterval=(90,0),maskcoords=(0,0),maskRec=(104,148),output="lena2_preview3.png")
    
