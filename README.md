Tracking Particle Flow with OpenPIV
This project includes different implementations using OpenPIV to track the motion of particle flow.

Overview:
The Compute_vector_field_CPU function calculates vector fields based on frames using OpenPIV, an open-source software library for Particle Image Velocimetry (PIV) analysis. The core idea is to split images into smaller regions called interrogation windows and use cross-correlation to determine particle displacement within each window. The velocity vectors are derived from the displacement and the time interval between frames.

Links to OpenPIV:
- OpenPIV Python: https://github.com/OpenPIV/openpiv-python.git
- OpenPIV Python GPU: https://github.com/OpenPIV/openpiv-python-gpu.git

  
Running the Script:
To run the script, specify the file format as follows:

    python3 piv_final.py binary_frames.npz

The script accepts the following input types:

- .h5/hdf5 files: The script will convert the data into frames.
- .npz files: Directly use the frames stored in this format.
- .tif files: You can also provide a directory containing .tif files.

The script calculates vector fields between each consecutive frame. Parameters can be adjusted, but the recommended values for optimal results are:

- winsize = 32
- searchsize = 38
- overlap = 16
- dt = 0.02

These parameters maintain a good balance between vector field resolution and computational efficiency. Lower values increase the number of vectors but also raise computational time.

The density threshold for the low-density mask may need to be adjusted depending on the parameters. For example, with:

- winsize = 64
- searchsize = 76
- overlap = 32
- dt = 0.02

A suitable density threshold might be density_threshold = 30.

Parallelization:
Both versions use a parallelization script, but Compute_vector_field_CPU_c uses a C script for the save parts, which may offer better performance in certain scenarios depending on the hardware. In some cases, this version could be faster than the other implementation.

Real-Time Version:
The real-time version aims to compute and display frames in real time. Currently, it achieves this by following a 100 ms window and skipping 2 out of every 3 frames.

GPU Version:
For the GPU version, the equivalent parameters to winsize = 32, searchsize = 38, overlap = 16, and dt = 0.02 are:

```python
params = {
    'mask': None,
    'window_size_iters': ((192, 1), (96, 2)),
    'overlap_ratio': 0.5,
    'dt': 1,
    'num_validation_iters': 1,
    'validation_method': 'median_velocity',
}


Frame_into_video.py : 
If you need to display the frames as a video. Fps can be adjusted. It convert frame in opencv format because it is easely to handle it whith opencv.

