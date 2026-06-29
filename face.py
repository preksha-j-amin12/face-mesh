import cv2
import mediapipe as mp

# Initialize MediaPipe Face Mesh and Drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# 0 is the default built-in webcam. Try 1 or 2 if a window doesn't appear.
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam at index 0.")
    exit()

# Configure the Face Mesh model
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,  # Set to True for eye iris tracking
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:
    
    print("Press 'ESC' on the video window to exit.")
    empty_frame_count = 0
    
    while cap.isOpened():
        success, image = cap.read()
        
        if not success:
            empty_frame_count += 1
            print(f"Warning: Empty camera frame detected ({empty_frame_count}).")
            # If it fails 10 times in a row, the camera index is likely wrong
            if empty_frame_count > 10:
                print("Error: Too many empty frames. Try changing cv2.VideoCapture(0) to 1 or 2.")
                break
            continue
        
        # Reset counter on a successful frame read
        empty_frame_count = 0

        # Optimize performance by marking the image as unwriteable
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)

        # Draw the face mesh annotations on the image
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
        
        # Display the output
        cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))
        
        # Increased waitKey to 30ms to ensure the Windows GUI has time to render
        if cv2.waitKey(30) & 0xFF == 27:
            break

# Clean up and close windows properly
cap.release()
cv2.destroyAllWindows()
print("Script closed safely.")
