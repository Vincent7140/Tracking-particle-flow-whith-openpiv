import numpy as np
import os
import matplotlib.pyplot as plt
import time
from openpiv import pyprocess, validation, filters
from scipy.ndimage import median_filter
from openpiv.gpu import process
import h5py
from multiprocessing import Pool
from PIL import Image
import sys


params = {
    'mask': None,
    'window_size_iters': ((96, 1), (48, 2)),
    'overlap_ratio': 0.5,
    'dt': 1,
    'num_validation_iters': 1,
    'validation_method': 'median_velocity',
}

def get_vector_field(frame1, frame2):
    x, y, u, v, mask, s2n = process.gpu_piv(frame1, frame2, **params)

    u, v = filters.replace_outliers(u, v, mask, method='localmean', max_iter=3, kernel_size=3)
    return x, y, u, v




# Argument check
if len(sys.argv) < 2:
    print("Usage: python script.py <input_file>")
    sys.exit(1)

input_file = sys.argv[1]
height, width = 720, 1280  # Frame dimensions



def load_tiff_frames(directory):
    frames = []
    for filename in sorted(os.listdir(directory)):
        if filename.endswith(".tif") or filename.endswith(".tiff"):
            filepath = os.path.join(directory, filename)
            image = Image.open(filepath)
            frames.append(np.array(image))
    return frames

directory = "Filt" # Put whatever directory you used to save the tiff images
binary_frames = load_tiff_frames(directory)
# print("size of first frame" , binary_frames[0].shape)
def chunk_events_by_time(t, time_window_ms):
    chunks = []
    start_idx = 0
    end_time = t[0] + time_window_ms * 1000
    for i in range(len(t)):
        if t[i] >= end_time:
            chunks.append((start_idx, i))
            start_idx = i
            end_time = t[i] + time_window_ms * 1000
    chunks.append((start_idx, len(t)))
    return chunks

def accumulate_events_in_chunk(x, y, t, p, start_idx, end_idx):
    frame_events = [(x[i], y[i], t[i], p[i]) for i in range(start_idx, end_idx)]
    return np.array(frame_events)

def convert_to_binary_frame(frame_events):
    binary_frame = np.zeros((height, width), dtype=np.uint8)
    for event in frame_events:
        x, y, _, p = event
        color = 255 if p == 1 else 0
        binary_frame[int(y), int(x)] = color
    return binary_frame

def process_chunk(chunk):
    start_idx, end_idx = chunk
    frame_events = accumulate_events_in_chunk(x, y, event_time, polarity, start_idx, end_idx)
    return convert_to_binary_frame(frame_events)

# Function to process NPZ file
def process_npz_file(input_file):
    with np.load(input_file) as data:
        binary_frames = [data[f"arr_{i}"] for i in range(len(data.files))]
    print("NPZ file loaded successfully.")
    return binary_frames

# Function to process TIFF files
def process_tiff_file(directory):
    frames = []
    for filename in sorted(os.listdir(directory)):
        if filename.endswith(".tif") or filename.endswith(".tiff"):
            filepath = os.path.join(directory, filename)
            image = Image.open(filepath)
            frames.append(np.array(image))  # Convert the image to a numpy array
    print(f"Loaded {len(frames)} TIFF frames from {directory}.")
    return frames



# Main logic to handle file formats
if input_file.endswith('.hdf5') or input_file.endswith('.h5'):
    with h5py.File(input_file, 'r') as f:
        events = f['CD/events'][:]
        event_time = events['t']
        x = events['x']
        y = events['y']
        polarity = events['p']
    time_window_ms = 20
    chunks = chunk_events_by_time(event_time, time_window_ms)
    start_time = time.time()
    with Pool() as pool:
        binary_frames = pool.map(process_chunk, chunks)
    end_time = time.time()

    print(f"Time to convert events into binary frames using parallel processing: {end_time - start_time:.4f} seconds")
    np.savez_compressed("binary_frames_all.npz", *binary_frames)
    print("Binary frames saved successfully.")
elif input_file.endswith('.npz'):
    binary_frames = process_npz_file(input_file)
elif os.path.isdir(input_file):
    start_time = time.time()

    binary_frames = process_tiff_file(input_file)

    end_time = time.time()

    print(f"Time to convert .tif/.tiff files: {end_time - start_time:.4f} seconds")

    np.savez_compressed("binary_frames_all.npz", *binary_frames)
    print("Binary frames saved successfully.")
else:
    print("Unsupported file format. Please provide a .npz or .hdf5/.h5 file.")
    sys.exit(1)

winsize = 32
searchsize = 38
overlap = 16
dt = 0.02

def compute_vector_field(index):
    frame1 = binary_frames[index]
    frame2 = binary_frames[index + 1]
    x, y, u, v = get_vector_field(frame1, frame2)
    return index, frame1, x, y, u, v

def real_time_display():
    start_time = time.time()
    fig, ax = plt.subplots(figsize=(10, 10))

    # Loop through every third frame
    for index in range(0, len(binary_frames) - 1, 3):
        frame1 = binary_frames[index]
        frame2 = binary_frames[index + 1]

        x, y, u, v = get_vector_field(frame1, frame2)

        ax.clear()

        ax.imshow(frame1, cmap='gray')
        ax.quiver(x, y, u, v, color='r')
        ax.set_title(f"Vector field between frame {index} and {index + 1}")

        plt.pause(0.01)  

    plt.close(fig)
    plt.close('all')
    total_processing_time = time.time() - start_time
    print(f"Total processing time: {total_processing_time:.4f} seconds")

# Call the parallelized real-time display function
real_time_display()