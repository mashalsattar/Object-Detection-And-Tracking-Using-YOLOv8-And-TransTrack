# Object-Detection-And-Tracking-Using-YOLOv8-And-TransTrack

This project implements a real-time object tracking system using the pre-trained DETR (DEtection Transformer) model from Facebook Research. It utilizes a webcam feed and detects objects within each frame.

This project uses a pre-trained DETR model for object detection, not YOLOv8. YOLOv8 is a separate object detection model that is not used here.

# Project Details

This project focuses on demonstrating real-time object tracking through a simple application. It is not intended for production use and lacks functionalities like custom object class detection or model training.

# Labeling Images

Since this project utilizes a pre-trained DETR model, image labeling is not required. The model is already trained on a large dataset of various objects.

# System Requirements

Python 3.6 or later

A GPU is recommended for faster performance, but the code can run on CPU as well.

The aforementioned libraries need to be installed using pip installopencv-python torch torchvision Pillow numpy.
Running the Project

# Install the required libraries.

pip install OpenCV (cv2)

pip install PyTorch

pip install Torchvision

pip install Pillow (PIL Fork)

pip install NumPy

# Labeling Dataset with LabelImg
If you need to label your dataset, you can use LabelImg, an open-source graphical image annotation tool.

Using LabelImg to Label Your Dataset
Install LabelImg:

```
pip install labelImg
```
Run LabelImg:
```
labelImg
```
**Label your images:**

Open your images in LabelImg.

Draw bounding boxes around the objects of interest.

Save the annotations in the Pascal VOC format.

# Training YOLOv8 :

**Prepare your dataset:** You'll need a labeled dataset in the COCO format. Tools like VGG Image Annotator (VIA) or LabelImg can be used for labeling.

**Dataset Structure**

**YOLOv8 typically requires the following directory structure:**
```
/dataset
    /images
        /train
 image1.jpg
            image2.jpg
            ...
        /val
            image1.jpg
            image2.jpg
            ...
    /labels
       /train
            image1.txt
            image2.txt
            ...
        /val
 image1.txt
            image2.txt
            ...
```
**Images:** JPEG or PNG files for training and validation.

**Labels:** YOLO format text files (.txt) for annotations 

**Train the model:** Refer to the YOLOv8 documentation (https://github.com/computervisioneng/train-yolov8-custom-dataset-step-by-step-guide) for detailed training instructions. The general steps involve creating a YAML configuration file, preparing your dataset, and running the training script.

# Train the Model

With YOLOv8 installed, you can train your model using the following command:
```
yolo train model=yolov8n.pt data=data.yaml epochs=20 imgsz=100
```
**Parameters Explained:**

**model=yolov8n.pt:** The pre-trained YOLOv8 model to start with (YOLOv8 has different versions like yolov8n for nano, yolov8s for small, etc.).

**data=data.yaml:** Path to your YAML configuration file.

**epochs=20:** Number of training epochs.

**imgsz=100:** Image size (resolution) for training.

# F1 Curve:

![F1_curve](https://github.com/user-attachments/assets/7c26e41e-42f8-4941-8840-a7080327ad70)

# Monitor Training

The training process will generate logs and save checkpoints. Monitor the output for metrics like loss, precision, recall, and mAP (mean Average Precision). You can visualize the training progress using tools like TensorBoard or directly from the log files.

# Confusion Matrix:

![confusion_matrix](https://github.com/user-attachments/assets/75228404-df03-4cb0-b76c-ed2a057f329d)

# Annotation Formats
LabelImg supports multiple annotation formats, but Pascal VOC is widely used for object detection tasks.

# Flowchart Of The Process:

![WhatsApp Image 2024-08-10 at 2 24 55 PM](https://github.com/user-attachments/assets/8abe34a9-fb29-40ca-aaaf-16e99ec5aeb9)

# Results:

![results](https://github.com/user-attachments/assets/7e20e9fe-8bee-42d8-8e33-c60352d0d02f)

# RealTime Results:

![Output](https://github.com/user-attachments/assets/5f9b3357-604f-474e-8665-c6f422ef1806)

# Disclaimer

The pre-trained DETR model used in this project might not detect all objects perfectly. Its performance depends on the training data and specific object types.
