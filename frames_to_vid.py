# Create a script that reads a dir with some frames and converst them to a video
import cv2
import os

# Path to the directory containing frames
path = '/mnt/e/13_Jasper_diffused_samples/complete_dataset/nuscenes_frames/A_not_linked/scene-0005'

# Get list of all files in the folder
files = os.listdir(path)

def numerical_sort(file_list):
    """Sorts the given list of files numerically based on the digits in the filenames."""
    # Extract the numerical part of each filename and use it as the key for sorting
    file_list.sort(key=lambda x: int(x.split('frame')[-1].split('.png')[0]))
    
    return file_list


# Filter and sort files to ensure they are in the correct order
frames = numerical_sort([f for f in files if f.endswith(".png") or f.endswith(".jpg") or f.endswith(".jpeg")])[:16]

# Read the first frame to determine the size
frame = cv2.imread(os.path.join(path, frames[0]))
height, width, layers = frame.shape

# Create a VideoWriter object
video = cv2.VideoWriter('video.avi', cv2.VideoWriter_fourcc(*'DIVX'), 1, (width, height))

# Iterate over sorted frames
for frame in frames:
    # Read each frame
    img = cv2.imread(os.path.join(path, frame))
    # Check if the frame was correctly loaded
    if img is not None:
        video.write(img)
    else:
        print(f"Warning: Could not read frame {frame}")

# Release the VideoWriter object
video.release()

print("The video was successfully created.")
