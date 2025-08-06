
import os
import cv2

def extract_first_frames(input_dir):
    """
    提取指定目录下所有MP4视频的首帧并保存为PNG
    :param input_dir: 视频文件所在目录路径
    """
    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.mp4'):
            video_path = os.path.join(input_dir, filename)
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                print(f"无法打开视频文件: {video_path}")
                continue
            
            ret, frame = cap.read()
            if ret:
                output_path = os.path.join(input_dir, 
                                         os.path.splitext(filename)[0] + '.png')
                cv2.imwrite(output_path, frame)
                print(f"已提取: {output_path}")
            else:
                print(f"无法读取视频首帧: {video_path}")
            
            cap.release()

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("用法: python extract_first_frame.py <视频目录路径>")
        sys.exit(1)
    
    extract_first_frames(sys.argv[1])
