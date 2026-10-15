import cv2
import os
import argparse

args = argparse.ArgumentParser()
args.add_argument('--name', type=str, default='frame')
args.add_argument('--video_path', type=str, default='test.mp4')
args.add_argument('--output_folder', type=str, default='vid_extract')
args.add_argument('--step', type=int, default=50)
args = args.parse_args()

# extract_list = [412,3080,1156,2425,2765,3695,5945,6050,6650,8290,8480,8600,8950,9100,9500,9560,9845,10110,10200]
# extract_list = [2675,2680,2670,2685]
# extract_list = [2150]

def extract_frames(video_path, output_folder, step=25, args=None):
    # Create output folder if it doesn't exist
    output_folder = video_path.split('/')[-1] 
    os.makedirs(output_folder, exist_ok=True)
    
    print(output_folder)
    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Initialize variables
    frame_count = 0
    saved_frame_count = 24

    while True:
        ret, frame = video.read()
        if not ret:
            break
        frame = frame[:,320:1600]
        # Save the frame if it's the desired frame
        # if frame_count in extract_list:
        if frame_count % step == 0:
            output_path = os.path.join(output_folder, f"{args.name}_{saved_frame_count:04d}.png")
            cv2.imwrite(output_path, frame)
            saved_frame_count += 1

        frame_count += 1

    # print progress
    print(f"Extracted {saved_frame_count} frames from {frame_count} frames")

    # Release the video file
    video.release()


if __name__ == '__main__':
    extract_frames(args.video_path, args.output_folder, args.step, args)

