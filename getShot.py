import cv2
from PIL import Image
import imagehash
class VideoReadError(Exception):
    pass


def calculate_image_hash(img):
    return imagehash.average_hash(img)


def is_image_similar(img1, img2, threshold=15):
    img1 = Image.fromarray(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    img2 = Image.fromarray(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))

    img1_hash = calculate_image_hash(img1)
    img2_hash = calculate_image_hash(img2)
    return img1_hash - img2_hash < threshold


def slice_video(src,startframeCtn,endframeCtn):
    video = cv2.VideoCapture(src)
    total=video.get(7)
    if endframeCtn>=total:
        endframeCtn=total-1
    if not video.isOpened():
        raise VideoReadError(f'视频文件读取失败，请检查 {src}')
    video.set(1,startframeCtn)
    start_frame_no = startframeCtn
    ret, start_frame = video.read()
    clip_duration_list = []
    ctn=startframeCtn
    while ctn<=endframeCtn:
        ret, frame = video.read()
        if not ret:
            break
        if not is_image_similar(start_frame, frame):
            current_frame_no = video.get(cv2.CAP_PROP_POS_FRAMES)
            clip_duration_list.append((start_frame_no, current_frame_no-2))
            start_frame_no = current_frame_no-1
        start_frame = frame
        ctn+=1
    clip_duration_list.append((start_frame_no,endframeCtn))

    return clip_duration_list


if __name__ == "__main__":

    print(slice_video('lena2.mp4',40,170))