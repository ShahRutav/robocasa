import h5py
import numpy as np
import cv2

# go through output hdft5 file and get the video list with camera anem
# output_hdf5_path = "/Users/rutavms/research/gaze/icrt_private/test_putbreadmicrowave/2025-07-24-16-15-57/demo_im128_notp.hdf5"
output_hdf5_path = "/Users/rutavms/research/gaze/icrt_private/test_washandreturn/2025-07-24-16-22-53/demo_im128_notp.hdf5"
video_file = output_hdf5_path.replace(".hdf5", ".mp4")
camera_names = ["robot0_agentview_center_image", "robot0_eye_in_hand_image"]
output_hdf5 = h5py.File(output_hdf5_path, "r")
video_list = []
data = output_hdf5["data"]
data_keys = list(data.keys())
data = data[data_keys[0]]

video_np = np.concatenate(
    [data["obs/{}".format(camera_name)] for camera_name in camera_names], axis=1
)
output_hdf5.close()

# Get video dimensions from the first frame
height, width = video_np[0].shape[:2]

# Create video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # or 'XVID' for .avi
video_writer = cv2.VideoWriter(video_file, fourcc, 30.0, (width, height))  # 30 fps

# Write each frame to the video
for i in range(video_np.shape[0]):
    frame = video_np[i]
    # Ensure frame is in BGR format (OpenCV expects BGR)
    if len(frame.shape) == 3 and frame.shape[2] == 3:
        # If it's RGB, convert to BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    video_writer.write(frame)

# Release the video writer
video_writer.release()
