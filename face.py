import cv2
import mediapipe as mp
import sys

print("1. Initializing MediaPipe modules...")
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# Try index 0 first. If the window closes right away, change this to 1 or 2.
camera_index = 0
print(f"2. Connecting to webcam index {camera_index}...")
cap = cv2.VideoCapture(camera_index)

if not cap.isOpened():
    print(f"Error: Could not open webcam source at index {camera_index}.")
    sys.exit()

print("3. Instantiating Face Mesh object...")
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

print("4. Setup complete. Starting video loop... (Press ESC on the window to exit)")

empty_frame_count = 0

try:
    while cap.isOpened():
        success, image = cap.read()
        
        if not success:
            empty_frame_count += 1
            print(f"Warning: Received an empty frame from camera ({empty_frame_count}/10).")
            # Force close if the camera is active but sending bad data
            if empty_frame_count > 10:
                print("\nError: Too many consecutive empty frames. Your webcam might be busy or blocked.")
                print(f"Try changing camera_index = 0 to 1 on line 9.")
                break
            cv2.waitKey(100) # Give the hardware a moment to recover
            continue
        
        # Reset counter on a successful frame read
        empty_frame_count = 0

        # Flip the image horizontally for a mirror view, convert to RGB
        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        
        results = face_mesh.process(image_rgb)

        # Allow drawing on the original image
        image.flags.writeable = True

        if results.multi_face_landmarks:
            mesh_spec = mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=1, circle_radius=1)
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=mesh_spec,
                    connection_drawing_spec=mesh_spec
                )

        # Show the final frame
        cv2.imshow('MediaPipe Face Mesh', image)

        # 30ms delay handles Windows GUI scaling loops smoothly
        if cv2.waitKey(30) & 0xFF == 27:
            print("ESC key detected. Exiting program.")
            break

except Exception as e:
    print(f"\nAn error occurred during runtime: {e}")

finally:
    print("Cleaning up resources...")
    face_mesh.close()
    cap.release()
    cv2.destroyAllWindows()
    print("Program completely shut down.")
