import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import mediapipe as mp

# Configure the mobile browser meta headers for PWA capabilities
st.markdown(
    """
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-status-bar-style" content="black">
    <link rel="manifest" href="./manifest.json">
    """,
    unsafe_allow_html=True
)

st.title("📱 Real-Time Face Mesh App")
st.write("Click **Start** below to allow camera access and run the mesh tracking directly on your phone.")

# Initialize MediaPipe Face Mesh modules
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

class FaceMeshTransformer(VideoTransformerBase):
    def __init__(self):
        # Instantiate Face Mesh object similarly to your original script
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def transform(self, frame):
        # Convert the incoming browser frame to a standard numpy array
        image = frame.to_ndarray(format="bgr24")
        
        # Mirror image for natural view, exactly like your original script
        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        results = self.face_mesh.process(image_rgb)

        # Draw the 468-point mesh annotations if a face is detected
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
        return image

# Streamlit-WebRTC component handles the mobile browser webcam layout
webrtc_streamer(
    key="face-mesh-stream", 
    video_transformer_factory=FaceMeshTransformer,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }
)
