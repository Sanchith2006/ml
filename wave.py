import cv2
import matplotlib.pyplot as plt
import numpy as np

# Constants
cap = cv2.VideoCapture(0)  # Video capture from the default camera (change the index if needed)
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

# Main loop for real-time video capture and plotting time waveform
while True:
    ret, frame = cap.read()  # Read a frame from the video capture
    
    # Display the video frame
    cv2.imshow('Video', frame)
    
    # Plot the time waveform
    plot_time_waveform(frame)
    
    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
