# Delogo

## 简介

这是一个消除一个视频中logo的工具。

基于开源库  **Flow-edge Guided Video Completion** 【FGVC】我们可以输入一段视频和一段mask就可以将mask中的内容补出来。

为了优化去logo场景下的速度和资源占用，我们采用了一系列方法包括裁剪，缩放，抽帧和AI插帧还原【RIFE】等。

## 环境配置

```shell
cd FGVC
conda create -n yourname
conda activate yourname
conda install pytorch=1.6.0 torchvision=0.7.0 cudatoolkit=10.1 -c pytorch
(pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html)
conda install matplotlib scipy
pip install -r requirements.txt
chmod +x download_data_weights.sh
./download_data_weights.sh
cd ../RIFE
pip install -r requirements.txt
pip install imagehash
```

## 参数说明

```python
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
```

## 接口调用

#### 函数接口

```python
delogo=Delogo() 
#for video
delogo.run(path="lena2.mp4",captionMode=True,extractFlag=False,frameInterval=(0,18),maskcoords=(138,424),maskRec=(337-138,493-424),output="lena2_out_caption.mp4") 
#for preview
delogo.preview(path="lena2.mp4",frameInterval=(0,0),maskcoords=(138,424),maskRec=(337-138,493-424),output="lena2_preview.png",captionMode=True)
```

#### 命令行接口

```
python removeLogo.py --path lena2.mp4 --maskcoords "(0,423)" --maskRec "(344,490)" --frameInterval "(0,18)" --captionMode
```


## 优化思想

##### cut和resize

如果将整个视频都拿到FGVC中进行处理，不仅计算缓慢，而且占用内存和现存资源极大。我们只将logo处一小块儿正方形截取出来，再通过缩放成统一的160*160的大小，再送入FGVC中处理。

##### shot和patch

由于FGVC进行计算的时候不仅依赖空间信息还依赖前后帧的时间信息，如果不进行shot切割，不仅计算量十分巨大，而且不能多线程加速。通过可选的是否将shot进行进一步的小段的切割，开发者可选择效果优先还是效率优先。

##### extract

为了进一步优化处理时间，我们可以每隔一帧抽取一帧，将这个视频序列放在FGVC中处理，得到的视频序列再通过AI补帧恢复到原来的帧数。可以选择是否进行抽帧决定效率优先还是效果优先。

##### 并行

因为将视频分割成了shot或者patch，我们可以采用并行进行加速。FGVC和RIFE(补帧)都用了并行进行加速。

![image-20210720174545067](note2.assets/image-20210720174545067.png)

## 处理过程

#### 长视频

###### 1、初始化参数  initGlobalVeriable

- 初始化mask和cut的坐标和宽高
- 初始化mask图像
- 初始化frame的宽高
- 初始化FGVC和RIFE的model

###### 2、shot切割   slice_video

将整个视频切成一个一个的shot，得到一个shotLs

比如：

```
shotLs=[(0, 65), (66, 92), (93, 131), (132, 157), (158, 291), (292, 362)]
```

###### 2、patch分割  getSplitLs

将每个shot分成大约30帧的patch

```
finalLs=[(0, 32), (33, 65), (66, 92), (93, 131), (132, 157), (158, 191), (192, 225), (226, 259), (260, 291), (292, 327), (328, 362)]
```

###### 3、得到cut的去水印视频  getcompletion

每隔一帧抽取一帧得到视频帧序列，调用FGVC函数，得到一段一段视频

###### 4、插帧 getRIFE

将一段一段视频通过AI算法进行插帧恢复到原来的帧数

###### 5、叠加恢复 getOverlayVideo

将处理之后的cut视频叠加到原视频上，得到最终的视频

### 单个片段

对于单个片段就不需要shot和patch的分割，直接就是 `finalLs=[(startFrameCtn,endFrameCtn)]`

## 参数建议

如果logo在左上角，尽量让mask覆盖整个左上角区域，也就是maskCoords=(0,0)。这样处理视频和原视频的叠加不会特别突兀。
