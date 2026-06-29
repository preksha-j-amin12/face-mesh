import cv2
import mediapipe as mp
import sys

print("1. Initializing MediaPipe modules...")
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

print("2. Connecting to webcam index 0...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam source at index 0. Try changing to 1 or 2.")
    sys.exit()

print("3. Instantiating Face Mesh object directly...")
# Using direct assignment instead of a 'with' block to isolate the crash point
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

print("4. Setup complete. Starting video loop... (Press ESC on the window to exit)")

try:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Warning: Received an empty frame from camera.")
            continue

        # Convert image color spaces for processing
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = face_mesh.process(image)

        # Revert changes to render annotations
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

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

        # Show the processed image
        cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))

        # Wait 30ms for a key press
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
