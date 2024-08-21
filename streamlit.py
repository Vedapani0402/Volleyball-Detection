#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile


def Detection_Tracking(upload_video):
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(upload_video.read())

    # Output video file
    output_video_path = 'input video path'

    # Load the YOLOv8 model
    model = YOLO('detection model best.pt weights path')

    # Open the input video
    cap = cv2.VideoCapture(tfile.name)

    # Get video frame dimensions
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    # Define the codec and create VideoWriter object to write the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You may need to change the codec based on your system and file format
    out = cv2.VideoWriter(output_video_path, fourcc, 30, (frame_width, frame_height))

    while cap.isOpened():
           # Read a frame from the video
            success, frame = cap.read()

            if success:
                      # Run YOLOv8 tracking on the frame, persisting tracks between frames
                     results = model.track(frame, persist=True, tracker="bytetrack.yaml")

                      # Visualize the results on the frame
                     annotated_frame = results[0].plot()


#                     # Display the annotated frame
#                      cv2_imshow(annotated_frame)

                      # Break the loop if 'q' is pressed
                     if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
            else:
                    # Break the loop if the end of the video is reached
                    break

    # Perform object detection on the frame (replace with your YOLO detection code)
    # Detected objects will have bounding boxes drawn on the frame
    # You should modify this part based on your YOLO implementation

    # Write the frame with bounding boxes to the output video
            out.write(annotated_frame)

# Release the video capture and writer objects
    cap.release()
    out.release()
# # Display the video using Streamlit
# st.video(output_video_path)

    with open(output_video_path, 'rb') as v:
         st.video(v)


# In[13]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow_docs.vis import embed
from tensorflow import keras
from imutils import paths

import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import imageio
import cv2
import keras
import os

def In_Out(upload_video):
    def build_feature_extractor():
        feature_extractor = keras.applications.InceptionV3(
               weights="imagenet",
               include_top=False,
               pooling="avg",
               input_shape=(224, 224, 3),
               )
        preprocess_input = keras.applications.inception_v3.preprocess_input

        inputs = keras.Input((224,224, 3))
        preprocessed = preprocess_input(inputs)

        outputs = feature_extractor(preprocessed)
        return keras.Model(inputs, outputs, name="feature_extractor")


    feature_extractor = build_feature_extractor()
    
    MAX_SEQ_LENGTH = 20
    NUM_FEATURES = 2048
    video_length=20


    def prepare_single_video(frames):
        frames = frames[None, ...]
        frame_mask = np.zeros(shape=(1, MAX_SEQ_LENGTH,), dtype="bool")
        frame_features = np.zeros(shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")

        for i, batch in enumerate(frames):
            video_length = batch.shape[0]
            length = min(MAX_SEQ_LENGTH, video_length)
            for j in range(length):
                frame_features[i, j, :] = feature_extractor.predict(batch[None, j, :])
            frame_mask[i, :length] = 1  # 1 = not masked, 0 = masked

        return frame_features, frame_mask

    def crop_center_square(frame):
        y, x = frame.shape[0:2]
        min_dim = min(y, x)
        start_x = (x // 2) - (min_dim // 2)
        start_y = (y // 2) - (min_dim // 2)
        return frame[start_y : start_y + min_dim, start_x : start_x + min_dim]


    def load_video(path, max_frames=0, resize=(224, 224)):
        cap = cv2.VideoCapture(path)
        frames = []
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = crop_center_square(frame)
                frame = cv2.resize(frame, resize)
                frame = frame[:, :, [2, 1, 0]]
                frames.append(frame)

                if len(frames) == max_frames:
                    break
        finally:
            cap.release()
        return np.array(frames)
    
    def get_sequence_model():
        class_vocab = ['In', 'Out']

        frame_features_input = keras.Input((MAX_SEQ_LENGTH, NUM_FEATURES))
        mask_input = keras.Input((MAX_SEQ_LENGTH,), dtype="bool")

        # Refer to the following tutorial to understand the significance of using `mask`:
        # https://keras.io/api/layers/recurrent_layers/gru/
        x = keras.layers.GRU(16, return_sequences=True)(frame_features_input, mask=mask_input)
        x = keras.layers.GRU(8)(x)
        x = keras.layers.Dropout(0.4)(x)
        x = keras.layers.Dense(8, activation="relu")(x)
        output = keras.layers.Dense(1, activation="sigmoid")(x)

        rnn_model = keras.Model([frame_features_input, mask_input], output)

        rnn_model.compile(
             loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
        )
        return rnn_model


    # Load the saved weights
    filepath = "video classifier weight path"
    # Recreate the model architecture
    seq_model= get_sequence_model()
    seq_model.load_weights(filepath)


    def sequence_prediction(path):
            class_vocab = ['In', 'Out']

            frames = load_video(os.path.join("test", path))
            frame_features, frame_mask = prepare_single_video(frames)
            probabilities = seq_model.predict([frame_features, frame_mask])[0]

            for i in np.argsort(probabilities)[::-1]:
                st.write(f"  {class_vocab[i]}: {probabilities[i] * 100:5.2f}%")
                st.write("out : ",f"{100-probabilities[i] * 100}%")
            return frames

    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(upload_video.read())
    test_frames = sequence_prediction(tfile.name)


# In[14]:


def main():
    
    st.sidebar.title('Select an Action to view')
    selected_action = st.sidebar.radio('Options : ',('Volleyball Detection and Tracking', 'In/Out Prediction'))

    if selected_action == 'Volleyball Detection and Tracking':
        st.title("Volleyball Detection And Tracking")
        uploaded_video1 = st.file_uploader("Upload video for Volleyball Tracking")
        if uploaded_video1 is not None:
            Detection_Tracking(uploaded_video1)
        else:
            st.warning("Please upload a video")  
    if selected_action == 'In/Out Prediction':
        st.title("In/Out Predictoin")
        uploaded_video1 = st.file_uploader("Upload video for In/Out Prediction")
        if uploaded_video1 is not None:
            In_Out(uploaded_video1)
        else:
            st.warning("Please upload a video") 

if __name__ == "__main__":
    main()





