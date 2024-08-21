# Volleyball-Detection and Tracking
 The purpose of this project is to detect and track the Volleyball in the input video. 

Initially, YOLOV8 pretrained model was used as inference. Based on the results, understood that Custom Training the YOLOV8 model on a dataset(videos) is necessary to achieve perfect Volleyball Detection and Tracking.

To create a dataset (so as to train YOLOV8 model), annotations are necessary( as YOLO model needs the bounding boxes for training on custom dataset). So Roboflow is used for Annotating the videos (frame by frame) to prepare a dataset for Custom Training of YOLOV8. Uploaded 10 videos with a condition of 20 frames per second which came around 800 images(roughly). The images are manually annotated, but Roboflow platform was simple, easy for annotations. To connect with the dataset created in Roboflow, Roboflow API is used.
The Dataset includes Train, Test and Validation data along with its labels (.txt files) and a yaml file to provide the dataset location for the YOLOV8 model. The model was trained on the annotated dataset with initial pretrained weights ‘yolov8s.pt’. The model was trained for 25 epochs keeping the image size of 800 and the ‘best.pt’ weight file is downloaded. The performance of the model is checked through some metrics like Confusion matrix, mAP50,mAP50-95 and some other line charts that define the loss, mAP etc.

The ‘best.pt’ weights were used in Realtime Inference. Since, the tracking of the detected ball in the video is important, the ‘bytetrack’ tracker provided by the YOLO model is used. During inference, the video is taken as an input, detection and tracking is done using the YOLOV8 with ‘best.pt’ weights for every frame, then each frame is added to create a ‘output_video’ video file with the ball detected in it throughout the video.

Along with Volleyball Detection, an In-Out Video Classification is trained which can can classify the input video whether in ball was inside the court or not.

## In-Out Classification
The model was trained on 500 videos with 450 videos under training and 50 videos under testing.For Video Classification, both Spatial and Temporal information of the video are important. So, a combined network is created with 2 main purposes:
     1. A feature extractor model to extract Spatial (Image based) Information and
     2. A Recurrent model for Temporal (Time based) information

For Feature Extractor, InceptionV3 is used and for Recurrent Model, a custom GRU network is created with output layer as a classification layer. So, the features learnt by InceptionV3 model are fed to the GRU model frame to frame and the output layer classifies the video.The trained model weights were used for Inference.

Finally, to view this, an interface is made with Streamlit.


NOTE : The best.pt weight file and the Video Classifier weight files are not uploaded in the repository. Remember that, for inference, those weight files were used.
       for Video Classifier, 3 files are required (checkpoint, video-classifier.index and video-classifier.data-***) for inference.
