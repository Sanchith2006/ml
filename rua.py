import cv2
import numpy as np
import matplotlib.pyplot as plt

# Constants for video capture
CAPTURE_WIDTH = 640
CAPTURE_HEIGHT = 480

# Initialize video capture
cap = cv2.VideoCapture(0)
cap.set(3, CAPTURE_WIDTH)
cap.set(4, CAPTURE_HEIGHT)

# Function to update the FFT plot in real-time
def update_plot(frame):
    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Perform FFT on the grayscale frame
    fft_data = np.fft.fft2(gray_frame)
    fft_shifted = np.fft.fftshift(fft_data)
    
    # Calculate magnitude spectrum
    magnitude_spectrum = np.abs(fft_shifted)
    
    # Update the plot with new data
    ax.clear()
    ax.imshow(np.log(1 + magnitude_spectrum), cmap='gray')
    ax.set_title('Real-time FFT Spectrum')
    plt.pause(0.001)

# Set up the plot
fig, ax = plt.subplots()

# Main loop for real-time video processing and visualization
while True:
    # Read a frame from the video capture
    ret, frame = cap.read()
    
    # Call the update_plot function with the video frame
    update_plot(frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the plot window
cap.release()
cv2.destroyAllWindows()
