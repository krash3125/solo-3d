import cv2
import os

video_path = "../vid_test.mp4"
output_dir = "test_vid_to_frames"

os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()


frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imwrite(f"{output_dir}/{frame_count}.jpg", frame)

    frame_count += 1

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
