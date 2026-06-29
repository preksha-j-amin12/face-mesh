import cv2
import mediapipe as mp

# Initialize MediaPipe Face Mesh and Drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# 0 is usually the default built-in webcam. Change to 1 if you have an external webcam.
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam. If you have an external camera, try changing cap = cv2.VideoCapture(0) to 1.")
    exit()

# Configure the Face Mesh model
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,  # Set to True for eye iris tracking
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:
    
    print("Press 'ESC' on the video window to exit.")
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Optimize performance by marking the image as unwriteable to pass by reference
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)

        # Draw the face mesh annotations on the image
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if results.multi_face_landmarks:
            # Custom style for the points/lines (Light blue/cyan mesh)
            mesh_spec = mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=1, circle_radius=1)
            
            for face_landmarks in results.multi_face_landmarks:
                # Draws the main facial tessellation
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=mesh_spec,
                    connection_drawing_spec=mesh_spec
                )
        
        # Display the output (flipped horizontally for a natural mirror effect)
        cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))
        
        # Break the loop when 'ESC' key (27) is pressed
        if cv2.waitKey(5) & 0xFF == 27:
            break

# Clean up and close windows properly
cap.release()
cv2.destroyAllWindows()
