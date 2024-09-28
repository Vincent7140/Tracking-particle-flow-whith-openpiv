import h5py
import numpy as np
import matplotlib.pyplot as plt
import time
from multiprocessing import Pool
import matplotlib.pyplot as plt
import matplotlib
import time
from scipy.ndimage import median_filter
import concurrent.futures
from openpiv import pyprocess, validation, filters
import time
import os
import sys
from PIL import Image
import numpy as np
import ctypes


# Argument check
if len(sys.argv) < 2:
    print("Usage: python script.py <input_file>")
    sys.exit(1)

input_file = sys.argv[1]
height, width = 720, 1280  # Frame dimensions

lib = ctypes.CDLL('./libsaveimage.so')

lib.save_image.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int, ctypes.c_int]
lib.save_image.restype = ctypes.c_int

###### For the conversion of hdf5 to binary frames ######
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

######    

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

with np.load("black_frame.npz") as data:
    black_frame = data['arr_0']









# Parameter for openPIV
winsize = 32
searchsize = 38
overlap = 16
dt = 0.02
fig, ax = plt.subplots(figsize=(20, 20))

## to remove vector when not enough particles
def compute_particle_density(frame, window_size, overlap):
    h, w = frame.shape
    y_range = range(0, h - window_size + 1, window_size - overlap)
    x_range = range(0, w - window_size + 1, window_size - overlap)
    
    density = np.zeros((len(y_range), len(x_range)))
    
    reduced_window_size = int(window_size * 0.7)
    offset = (window_size - reduced_window_size) // 1
    
    for i, y in enumerate(y_range):
        for j, x in enumerate(x_range):
            window = frame[y + offset:y + offset + reduced_window_size, x + offset:x + offset + reduced_window_size]
            density[i, j] = np.sum(window > 0)
    
   
    return density

def mask_low_density_regions(density, u, v,x,y, threshold):
    mask = density < threshold
    u[mask] = 0
    v[mask] = 0
    x[mask] = 0
    y[mask] = 0
    return u, v, x, y

def apply_median_filter(u, v, kernel_size=3, threshold_factor=1.5):
    u_median = median_filter(u, size=kernel_size)
    v_median = median_filter(v, size=kernel_size)
    
    u_diff = np.abs(u - u_median)
    v_diff = np.abs(v - v_median)
    
    u_threshold = threshold_factor * np.std(u)
    v_threshold = threshold_factor * np.std(v)
    
    u_outliers = u_diff > u_threshold
    v_outliers = v_diff > v_threshold
    
    u[u_outliers] = u_median[u_outliers]
    v[v_outliers] = v_median[v_outliers]
    
    return u, v


def get_vector_field(frame1, frame2,density_threshold=20): 
    u, v, sig2noise = pyprocess.extended_search_area_piv(
        frame1.astype(np.int32), 
        frame2.astype(np.int32), 
        window_size=winsize, 
        overlap=overlap, 
        dt=dt, 
        search_area_size=searchsize, 
        sig2noise_method='peak2peak'
    )

    x, y = pyprocess.get_coordinates(
        image_size=frame1.shape, 
        search_area_size=searchsize, 
        overlap=overlap
    )

    flags = validation.sig2noise_val(sig2noise, threshold=1.3)
    u, v = filters.replace_outliers(u, v, flags, method='localmean', max_iter=3, kernel_size=3)
    u, v = apply_median_filter(u, v, kernel_size=3, threshold_factor=1.5)

    density = compute_particle_density(frame1, window_size=searchsize, overlap=overlap)
    u, v, x, y = mask_low_density_regions(density, u, v, x, y, density_threshold)

    # print("u et v :", u, v)
    
    return x, y, u, v




# Fused function to compute, plot, and save vector field
def compute_plot_save_vector_field(index):
    frame1 = binary_frames[index]
    frame2 = binary_frames[index + 1]
    
    # Step 1: Compute vector field
    x, y, u, v = get_vector_field(frame1, frame2)
    
    # Step 2: Plot the vector field
    fig, ax = plt.subplots(figsize=(20, 20))
    ax.imshow(frame1, cmap='gray')
    ax.quiver(x, y, u, v, color='r')
    ax.set_aspect('equal')
    ax.set_title(f"Vector field between frame {index} and {index + 1}")
    
    fig.canvas.draw()

    # Convert plot to RGBA format (as a numpy array)
    width, height = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8).reshape(height, width, 4)
    
    # Convert ARGB to RGBA (as expected by libpng)
    buf = np.roll(buf, 3, axis=2)

    # Step 3: Save the figure using the C library
    directory = "vector_fields_c"
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    filename = os.path.join(directory, f"vector_field_{index}.png")
    
    # Call the C function to save the image
    result = lib.save_image(filename.encode('utf-8'), buf.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)), width, height)
    
    # Close the figure to free up memory
    plt.close(fig)
    
    if result != 0:
        print(f"Failed to save {filename}")
    return filename

# Step: Parallelize the fused computation, plotting, and saving
start_time = time.time()
with concurrent.futures.ProcessPoolExecutor(max_workers=12) as executor:
    vector_field_filenames = list(executor.map(compute_plot_save_vector_field, range(len(binary_frames) - 1)))
total_computation_time = time.time() - start_time
print(f"Total computation time: {total_computation_time:.4f} seconds")






