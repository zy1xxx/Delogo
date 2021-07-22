import cv2

ls=[(0, 33), (34.0, 39.0), (40.0, 43.0), (44.0, 58.0), (59.0, 81.0), (82.0, 83.0), (84.0, 100.0), (101.0, 124.0), (125.0, 147.0), (148.0, 173.0), (174.0, 202.0), (203.0, 231.0), (232.0, 260.0), (261.0, 298.0), (299.0, 336.0), (337.0, 365.0), (366.0, 381.0)]
if __name__=='__main__':
    cap1=cv2.VideoCapture('tmp/test2.mp4')
    cap2=cv2.VideoCapture('tmp/test0.mp4')
    cap3=cv2.VideoCapture('test2.2.mp4')
    cap4=cv2.VideoCapture('test2_2X_50fps.mp4')
    frames_num1=cap1.get(7)
    frames_num2=cap2.get(7)
    frames_num3=cap3.get(7)
    frames_num4=cap4.get(7)
    print(frames_num1,frames_num2,frames_num3,frames_num4)
    sum=0
    for i in range(len(ls)):
        cap4=cv2.VideoCapture('tmp/video'+str(i)+'.mp4')
        n=cap4.get(7)
        print(i,n)
        sum+=n
    print("sum",sum)
    sum=0
    for i in range(len(ls)):
        cap4=cv2.VideoCapture('tmp/videoRIFE'+str(i)+'.mp4')
        n=cap4.get(7)
        print(i,n)
        sum+=n
    print("sum",sum)
    sum=0
    for index,item in enumerate(ls):
        n=item[1]-item[0]+1
        sum+=n
        print(index,n)
    print("sum",sum)
