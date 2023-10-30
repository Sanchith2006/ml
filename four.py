import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # Convert the frame to grayscale and then to float32
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # Perform DFT
    DFT = cv2.dft(gray_frame, flags=cv2.DFT_COMPLEX_OUTPUT)
    shift = np.fft.fftshift(DFT)

    row, col = gray_frame.shape
    center_row, center_col = row // 2, col // 2
    mask = np.zeros((row, col, 2), np.uint8)
    mask[center_row - 30:center_row + 30, center_col - 30:center_col + 30] = 1

    fft_shift = shift * mask
    fft_ifft_shift = np.fft.ifftshift(fft_shift)
    image_then = cv2.idft(fft_ifft_shift)

    # Compute the magnitude spectrum
    magnitude_spectrum = cv2.magnitude(image_then[:, :, 0], image_then[:, :, 1])

    # Normalize the magnitude spectrum for display
    magnitude_spectrum_normalized = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX)

    # Convert magnitude spectrum to uint8 type
    magnitude_spectrum_uint8 = magnitude_spectrum_normalized.astype(np.uint8)

    # Display the magnitude spectrum
    cv2.imshow('Magnitude Spectrum', magnitude_spectrum_uint8)

    # Display the original frame
    cv2.imshow('Original Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
