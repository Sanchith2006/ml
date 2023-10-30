import numpy as np
import cv2
import time
import matplotlib.pyplot as plt

cap = cv2.VideoCapture(0)

lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict(maxCorners=20,
                      qualityLevel=0.3,
                      minDistance=10,
                      blockSize=7)

trajectory_len = 40
detect_interval = 5
trajectories = []
frame_idx = 0

sampling_frequency = 30  # Number of frames per second for capturing video

# Function to plot the time waveform
def plot_time_waveform(frame):
    plt.clf()  # Clear the previous plot
    time_waveform = np.mean(frame, axis=1)  # Average pixel values along the width (assuming grayscale)
    plt.plot(time_waveform)  # Plot the time waveform
    plt.xlabel('Time (frames)')
    plt.ylabel('Average Pixel Value')
    plt.title('Time Waveform')
    plt.pause(0.001)  # Pause to allow the plot to update

while True:
    ret, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    DFT = cv2.dft(gray_frame, flags=cv2.DFT_COMPLEX_OUTPUT)
    shift = np.fft.fftshift(DFT)

    row, col = gray_frame.shape
    center_row, center_col = row // 2, col // 2
    mask = np.zeros((row, col, 2), np.uint8)
    mask[center_row - 30:center_row + 30, center_col - 30:center_col + 30] = 1

    fft_shift = shift * mask
    fft_ifft_shift = np.fft.ifftshift(fft_shift)
    image_then = cv2.idft(fft_ifft_shift)

    magnitude_spectrum = cv2.magnitude(image_then[:, :, 0], image_then[:, :, 1])

    magnitude_spectrum_normalized = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX)

    magnitude_spectrum_uint8 = magnitude_spectrum_normalized.astype(np.uint8)

    cv2.imshow('Magnitude Spectrum', magnitude_spectrum_uint8)
    cv2.imshow('Original Frame', frame)

    start = time.time()
    frame_gray = magnitude_spectrum_uint8
    img = frame.copy()

    if len(trajectories) > 0:
        img0, img1 = prev_gray, frame_gray
        p0 = np.float32([trajectory[-1] for trajectory in trajectories]).reshape(-1, 1, 2)
        p1, _st, _err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
        p0r, _st, _err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
        d = abs(p0 - p0r).reshape(-1, 2).max(-1)
        good = d < 1

        new_trajectories = []

        for trajectory, (x, y), good_flag in zip(trajectories, p1.reshape(-1, 2), good):
            if not good_flag:
                continue
            trajectory.append((x, y))
            if len(trajectory) > trajectory_len:
                del trajectory[0]
            new_trajectories.append(trajectory)
            cv2.circle(img, (int(x), int(y)), 2, (0, 0, 255), -1)

        trajectories = new_trajectories

        cv2.polylines(img, [np.int32(trajectory) for trajectory in trajectories], False, (0, 255, 0))
        cv2.putText(img, 'track count: %d' % len(trajectories), (20, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)

    if frame_idx % detect_interval == 0:
        m = np.zeros_like(frame_gray)
        m[:] = 255
        for x, y in [np.int32(trajectory[-1]) for trajectory in trajectories]:
            cv2.circle(m, (x, y), 5, 0, -1)
        p = cv2.goodFeaturesToTrack(frame_gray, mask=m, **feature_params)
        if p is not None:
            for x, y in np.float32(p).reshape(-1, 2):
                trajectories.append([(x, y)])

    frame_idx += 1
    prev_gray = frame_gray

    end = time.time()

    # Calculate FPS only if start and end times are different
    if start != end:
        fps = 1 / (end - start)
    else:
        fps = 0  # Set FPS to 0 if start and end times are the same

    cv2.putText(img, f"{fps:.2f} FPS", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Mask', m)
    cv2.imshow('Optical Flow', img)
    plot_time_waveform(frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
