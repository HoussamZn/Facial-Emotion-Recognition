import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import threading
import queue
import time
import tempfile
import os
import pandas as pd
import base64
from collections import deque
import matplotlib.pyplot as plt

class CameraStream:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 60)  # Set FPS to 15
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer size
        
        # Initialize frame queue
        self.q = queue.Queue(maxsize=2)
        self.stopped = False
        
    def start(self):
        # Start frame collection thread
        threading.Thread(target=self.update, daemon=True).start()
        return self
    
    def update(self):
        while True:
            if self.stopped:
                return
            
            # Clear queue to ensure latest frame
            while not self.q.empty():
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    break
                    
            ret, frame = self.cap.read()
            if ret:
                # Invert the frame
                frame = cv2.flip(frame, 1)  # Horizontal flip
                # frame = 255 - frame  # Invert colors
                
                # Add frame to queue
                if not self.q.full():
                    self.q.put(frame)
            time.sleep(0.001)  # Small delay to prevent CPU overload
    
    def read(self):
        return self.q.get() if not self.q.empty() else None
    
    def stop(self):
        self.stopped = True
        self.cap.release()

class FacialExpressionDetector:
    def __init__(self, model_path):
        # Initialize MediaPipe with optimized settings
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Load model
        self.model = load_model(model_path)
        
        # Emotion labels
        self.emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        
        # Emotion colors (BGR format for OpenCV)
        self.emotion_colors = {
            'Angry': (0, 0, 255),      # Red
            'Disgust': (0, 140, 255),  # Orange
            'Fear': (0, 255, 255),     # Yellow
            'Happy': (0, 255, 0),      # Green
            'Sad': (255, 0, 0),        # Blue
            'Surprise': (255, 0, 255), # Magenta
            'Neutral': (255, 255, 255) # White
        }
        
        # Initialize face detectors with optimized settings
        self.face_detection = self.mp_face_detection.FaceDetection(
            min_detection_confidence=0.3,  # Lower threshold for faster detection
            model_selection=0  # Use faster model
        )
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=10,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3,
            refine_landmarks=False  # Disable landmark refinement for speed
        )

    def preprocess_face(self, face):
        try:
            face = cv2.resize(face, (48, 48))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = face.astype('float32') / 255.0
            face = np.expand_dims(face, axis=-1)
            face = np.expand_dims(face, axis=0)
            return face
        except Exception as e:
            return None

    def predict_emotion(self, face):
        try:
            predictions = self.model.predict(face, verbose=0)
            emotion_idx = np.argmax(predictions[0])
            confidence = predictions[0][emotion_idx]
            return self.emotions[emotion_idx], confidence
        except Exception as e:
            return None, None

    def process_frame(self, frame, show_face_mesh=True, text_size=0.5, text_thickness=1):
        if frame is None:
            return None, []

        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_height, frame_width, _ = frame.shape
        
        # Process frame with MediaPipe
        detection_results = self.face_detection.process(rgb_frame)
        mesh_results = self.face_mesh.process(rgb_frame)
        
        face_data = []
        
        # Process detected faces
        if detection_results.detections:
            for idx, detection in enumerate(detection_results.detections):
                bbox = detection.location_data.relative_bounding_box
                x = max(0, int(bbox.xmin * frame_width))
                y = max(0, int(bbox.ymin * frame_height))
                w = min(int(bbox.width * frame_width), frame_width - x)
                h = min(int(bbox.height * frame_height), frame_height - y)
                
                if w > 0 and h > 0:
                    face = frame[y:y+h, x:x+w]
                    processed_face = self.preprocess_face(face)
                    
                    if processed_face is not None:
                        emotion, confidence = self.predict_emotion(processed_face)
                        
                        if emotion and confidence:
                            # Get color for this emotion
                            color = self.emotion_colors.get(emotion, (0, 0, 255))
                            
                            # Draw rectangle with emotion-specific color
                            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                            
                            # Draw a filled rectangle for text background
                            label = f"{emotion} ({confidence:.2f})"
                            text_size_px = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 
                                                         text_size, text_thickness)[0]
                            cv2.rectangle(frame, (x, y-text_size_px[1]-10), 
                                         (x+text_size_px[0], y), color, -1)
                            
                            # Draw text with better visibility
                            cv2.putText(frame, label, (x, y-5),
                                      cv2.FONT_HERSHEY_SIMPLEX, text_size, 
                                      (0, 0, 0), text_thickness)
                            
                            face_data.append({
                                'face_id': idx,
                                'emotion': emotion,
                                'confidence': confidence,
                                'position': (x, y, w, h)
                            })
        
        if show_face_mesh and mesh_results.multi_face_landmarks:
            for face_landmarks in mesh_results.multi_face_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    face_landmarks,
                    self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                        color=(255, 145, 245),
                        thickness=1,
                        circle_radius=1
                    ),
                    connection_drawing_spec=self.mp_drawing.DrawingSpec(
                        color=(175, 0, 134),
                        thickness=1,
                        circle_radius=1
                    )
                )
        
        return frame, face_data
    
    def change_model(self, model_path):
        self.model = load_model(model_path)

def create_horizontal_bar_chart(face_data, emotions):
    """Create a horizontal bar chart for current emotion percentages"""
    # Count emotions in current frame
    emotion_counts = {emotion: 0 for emotion in emotions}
    
    # Count each emotion from current face_data
    for face in face_data:
        emotion_counts[face['emotion']] += 1
    
    # Calculate percentages
    total = sum(emotion_counts.values())
    if total == 0:
        return None
        
    emotion_percentages = {emotion: (count / total) * 100 for emotion, count in emotion_counts.items()}
    
    # Create the horizontal bar chart
    fig, ax = plt.figure(figsize=(6, 3)), plt.axes()
    
    # Sort emotions by percentage
    sorted_emotions = sorted(emotion_percentages.items(), key=lambda x: x[1], reverse=True)
    labels = [item[0] for item in sorted_emotions]
    values = [item[1] for item in sorted_emotions]
    
    # Create colors list matching our emotion colors (but in RGB for matplotlib)
    colors = ['red', 'orange', 'yellow', 'green', 'blue', 'magenta', 'gray']
    emotion_color_map = dict(zip(emotions, colors))
    bar_colors = [emotion_color_map.get(emotion, 'gray') for emotion in labels]
    
    # Create the horizontal bar chart
    bars = ax.barh(labels, values, color=bar_colors)
    
    # Add percentage labels to the bars
    for bar in bars:
        width = bar.get_width()
        label_x_pos = width + 1
        ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.1f}%',
                va='center')
    
    ax.set_xlabel('Percentage')
    ax.set_title('Current Emotion Distribution')
    plt.tight_layout()
    
    return fig
def process_uploaded_image(detector, uploaded_image, show_face_mesh):
    try:
        # Convert the uploaded image to a format compatible with OpenCV
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # Process the image with larger text
        processed_image, face_data = detector.process_frame(image, show_face_mesh, 
                                                        text_size=1.0, text_thickness=2)
        
        return processed_image, face_data
    except Exception as e:
        st.error(f"Error processing uploaded image: {str(e)}")
        return None, []

def process_video_frame(detector, frame, show_face_mesh):
    try:
        # Process the frame with larger text
        processed_frame, face_data = detector.process_frame(frame, show_face_mesh, 
                                                         text_size=1.0, text_thickness=2)
        return processed_frame, face_data
    except Exception as e:
        st.error(f"Error processing video frame: {str(e)}")
        return None, []


def set_page_config():
    # Set page configuration including theme
    st.set_page_config(
        page_title="Multi-Face Emotion Recognition",
        page_icon="😀",
        layout="wide",
        initial_sidebar_state="expanded"
    )

# # Add this function to create a theme selector
# def add_theme_selector():
#     # Add theme selector to sidebar
#     st.sidebar.title("App Settings")
#     theme = st.sidebar.radio(
#         "Choose Theme",
#         options=["Light", "Dark"],
#         index=0  # Default to Light
#     )
    
#     # Apply theme
#     if theme == "Light":
#         # Light theme
#         st.markdown("""
#         <style>
#         .stApp {
#             background-color: #FFFFFF;
#             color: #000000;
#         }
#         .css-145kmo2 {  /* Sidebar */
#             background-color: #F0F2F6;
#         }
#         .stButton>button {
#             background-color: #4CAF50;
#             color: white;
#         }
#         </style>
#         """, unsafe_allow_html=True)
#     else:
#         # Keep the default dark theme or customize it
#         st.markdown("""
#         <style>
#         .stButton>button {
#             background-color: #4CAF50;
#             color: white;
#         }
#         </style>
#         """, unsafe_allow_html=True)

def main():
    # Configure page
    set_page_config()
    
    # Add theme selector to sidebar
    # add_theme_selector()
    st.title("Multi-Face Emotion Recognition")
    
    model_options = {
        'VGG16-fer13': './model_deep/vgg16_fer13_model.keras',
        'VGG16-ck+': './model_deep/vgg16_ck+_model.keras',
        'VGG19-fer13': './model_deep/vgg19_fer13_model.keras',
        'VGG19-ck+': './model_deep/vgg19_ck+_model.keras',
        'EfficientNet b7-fer13': './model_deep/efficientnet_fer13_model.keras',
        'EfficientNet b7-ck+': './model_deep/efficientnet_ck+_model.keras',
        'custum_cnn-fer13' : './model_deep/custom_cnn_fer13_model.keras',
        'custum_cnn-ck+' : './model_deep/custum_cnn_ck+_model.keras'
    }
    
    # Create tabs for different input methods
    tab1, tab2, tab3 = st.tabs(["Webcam", "Upload Image", "Upload Video"])
    
    with tab1:
        st.header("Live Webcam Detection")
        model_selection = st.selectbox("Select Model", options=list(model_options.keys()), key="webcam_model")
        model_path = model_options[model_selection]
        
        show_face_mesh = st.checkbox("Show Face Mesh", value=False, key="webcam_mesh")
        
        col1, col2 = st.columns(2)
        start_button = col1.button("Start Camera")
        stop_button = col2.button("Stop Camera")
        
        # Create two columns - one for video, one for chart
        webcam_col1, webcam_col2 = st.columns([2, 1])
        
        with webcam_col1:
            frame_placeholder = st.empty()
        
        with webcam_col2:
            chart_placeholder = st.empty()
            metrics_placeholder = st.empty()
        
        if start_button and not stop_button:
            try:
                detector = FacialExpressionDetector(model_path)
                
                # Initialize threaded camera stream
                stream = CameraStream().start()
                
                # Initialize emotion tracking
                emotion_history = deque(maxlen=30)  # Track last 30 frames
                
                run = True
                while run and not stop_button:
                    frame = stream.read()
                    
                    if frame is not None:
                        processed_frame, face_data = detector.process_frame(frame, show_face_mesh, 
                                                                        text_size=1.0, text_thickness=2)
                        
                        if processed_frame is not None:
                            # Display the processed frame
                            frame_placeholder.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), 
                                                   caption="Live Feed", use_container_width=True)
                            
                            # Update emotion history
                            frame_emotions = [face['emotion'] for face in face_data]
                            emotion_history.append(frame_emotions)
                            
                            
                            # Create and display horizontal bar chart
                            chart_fig = create_horizontal_bar_chart(face_data, detector.emotions)
                            if chart_fig:
                                chart_placeholder.pyplot(chart_fig)
                            
                            # Display detailed metrics
                            metrics = ""
                            for face in face_data:
                                metrics += f"Face {face['face_id']}: {face['emotion']} ({face['confidence']:.2f})\n"
                            metrics_placeholder.text(metrics)
                    
                    # Check if stop button was pressed
                    if stop_button:
                        run = False
                        
                    time.sleep(0.001)  # Minimal sleep to prevent CPU overload
                
                stream.stop()
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    with tab2:
        st.header("Image Upload Detection")
        upload_model_selection = st.selectbox("Select Model", options=list(model_options.keys()), key="upload_model")
        upload_model_path = model_options[upload_model_selection]
        
        upload_show_face_mesh = st.checkbox("Show Face Mesh", value=False, key="upload_mesh")
        
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="image_uploader")
        
        if uploaded_file is not None:
            try:
                detector = FacialExpressionDetector(upload_model_path)
                
                # Process the uploaded image
                image_placeholder = st.empty()
                upload_metrics_placeholder = st.empty()
                
                # Show the original image
                st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
                
                # Process button
                if st.button("Process Image"):
                    # Reset the file pointer
                    uploaded_file.seek(0)
                    
                    # Process the image
                    processed_image, face_data = process_uploaded_image(detector, uploaded_file, upload_show_face_mesh)
                    
                    if processed_image is not None:
                        # Create columns for image and chart
                        img_col1, img_col2 = st.columns([2, 1])
                        
                        with img_col1:
                            # Display the processed image
                            img_col1.image(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB), 
                                          caption="Processed Image", use_container_width=True)
                        
                        with img_col2:
                            # Create emotion counts from single image
                            if face_data:
                                emotion_counts = {emotion: 0 for emotion in detector.emotions}
                                for face in face_data:
                                    emotion_counts[face['emotion']] += 1
                                
                                # Create and display horizontal bar chart
                                chart_fig = create_horizontal_bar_chart(emotion_counts, detector.emotions)
                                if chart_fig:
                                    img_col2.pyplot(chart_fig)
                                
                                # Display metrics
                                metrics = ""
                                for face in face_data:
                                    metrics += f"Face {face['face_id']}: {face['emotion']} ({face['confidence']:.2f})\n"
                                upload_metrics_placeholder.text(metrics)
                            else:
                                upload_metrics_placeholder.warning("No faces detected in the image.")
                    else:
                        st.error("Failed to process the image.")
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    with tab3:
        st.header("Video Upload Detection")
        video_model_selection = st.selectbox("Select Model", options=list(model_options.keys()), key="video_model")
        video_model_path = model_options[video_model_selection]
        
        video_show_face_mesh = st.checkbox("Show Face Mesh", value=False, key="video_mesh")
        
        # Add processing speed slider (frames to skip)
        processing_speed = st.slider("Processing Speed (1=fastest, 5=most accurate)", 1, 5, 2, key="video_processing_speed")
        frames_to_skip = 6 - processing_speed  # Convert slider to frame skip rate
        
        # Set a history window for the emotion tracking
        history_window = st.slider("Emotion History Window (frames)", 10, 100, 30, key="emotion_history_window")
        
        uploaded_video = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"], key="video_uploader")
        
        if uploaded_video is not None:
            # Display file info
            file_details = {"Filename": uploaded_video.name, "FileType": uploaded_video.type, "FileSize": f"{uploaded_video.size / (1024 * 1024):.2f} MB"}
            st.write(file_details)
            
            # Process video button
            if st.button("Process Video"):
                try:
                    detector = FacialExpressionDetector(video_model_path)
                    
                    # Save uploaded video to a temporary file
                    tfile = tempfile.NamedTemporaryFile(delete=False)
                    tfile.write(uploaded_video.read())
                    video_path = tfile.name
                    tfile.close()
                    
                    # Open the video file
                    cap = cv2.VideoCapture(video_path)
                    if not cap.isOpened():
                        st.error("Could not open video file")
                    else:
                        # Get video properties
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        
                        # Create two columns - one for video, one for chart
                        video_col1, video_col2 = st.columns([2, 1])
                        
                        # Placeholders for video and chart
                        with video_col1:
                            video_placeholder = st.empty()
                        with video_col2:
                            chart_placeholder = st.empty()
                            metrics_placeholder = st.empty()
                        
                        # Initialize emotion tracking
                        emotion_history = deque(maxlen=history_window)
                        frame_count = 0
                        
                        # Create a progress bar
                        progress_bar = st.progress(0)
                        
                        # Process frames
                        while cap.isOpened():
                            ret, frame = cap.read()
                            if not ret:
                                break
                                
                            # Process every Nth frame based on processing speed
                            if frame_count % frames_to_skip == 0:
                                # Process the frame
                                processed_frame, face_data = process_video_frame(detector, frame, video_show_face_mesh)
                                
                                if processed_frame is not None:
                                    # Display the processed frame
                                    video_placeholder.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB),
                                                        caption="Video Processing", use_container_width=True)
                                    
                                    # Update emotion history
                                    frame_emotions = [face['emotion'] for face in face_data]
                                    emotion_history.append(frame_emotions)
                                    
                                    # Create and display horizontal bar chart
                                    chart_fig = create_horizontal_bar_chart(face_data, detector.emotions)
                                    if chart_fig:
                                        chart_placeholder.pyplot(chart_fig)
                                    
                                    # Display metrics
                                    metrics = "Current emotions:\n"
                                    for face in face_data:
                                        metrics += f"Face {face['face_id']}: {face['emotion']} ({face['confidence']:.2f})\n"
                                    metrics_placeholder.text(metrics)
                            
                            # Update progress bar
                            progress_percent = min(frame_count / total_frames, 1.0)
                            progress_bar.progress(progress_percent)
                            
                            frame_count += 1
                            
                            # Add a small delay to make the video playback more visible
                            time.sleep(1/fps if fps > 0 else 0.025)
                        
                        # Release resources
                        cap.release()
                        os.unlink(video_path)
                        
                        st.success("Video processing complete!")
                        
                except Exception as e:
                    st.error(f"Error processing video: {str(e)}")
                
    # Add information about the app
    st.sidebar.title("About")
    st.sidebar.info(
        "This application detects facial expressions in real-time using "
        "deep learning models. Choose between webcam input, uploaded images, or videos."
    )
    
    # Add model information
    st.sidebar.title("Model Information")
    st.sidebar.markdown(
        """
        **Available Models:**
        - **VGG16**: Good balance of accuracy and speed
        - **VGG19**: Higher accuracy but slower
        - **EfficientNet b7**: State-of-the-art performance

        **Dataset Trained on:**
        - **FER 2013**: up to **35887 images** from the original FER 2013 dataset, 48 x 48 pixels, 7 classes, grayscale
        - **CK+**: up to **920 images** from the original CK+ dataset, 48 x 48 pixels, 7 classes, grayscale, face-cropped

        **Emotions Detected:**
        - Angry
        - Disgust
        - Fear
        - Happy
        - Sad
        - Surprise
        - Neutral
        """
    )

if __name__ == '__main__':
    main()