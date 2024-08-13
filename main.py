import cv2
import torch
from ultralytics import YOLO
from PIL import Image
import torchvision.transforms as T

# Uncomment and import TransTrack if available
# from transtrack import TranStrack  # Uncomment if TransTrack is available

# Configuration
USE_DETR = False  # Set to True to use DETR, False to use YOLOv8
YOLO_WEIGHT_PATH = 'C:\\Users\\user\\Desktop\\Object-Detection-And-Tracking-Using-YOLOv8-And-TransTrack\\dataset\\weights\\bottle.pt'
DETECTION_CONFIDENCE_THRESHOLD = 0.8  # Minimum confidence for detection

# Load the models
if USE_DETR:
    model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)
    model.eval()

    # Transform for the input frame
    transform = T.Compose([
        T.Resize(800),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
else:
    model = YOLO(YOLO_WEIGHT_PATH)

# Tracker variable to be initialized later
tracker = None
tracker_initialized = False
object_bbox = None

def get_bounding_boxes_detr(outputs, threshold=0.7):
    """ Get bounding boxes from DETR model output """
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > threshold
    boxes = outputs['pred_boxes'][0, keep]
    labels = probas.argmax(-1)[keep]
    confidences = probas.max(-1).values[keep]
    return boxes, labels, confidences

def rescale_bboxes(out_bbox, size):
    """ Rescale bounding boxes to the original image size """
    img_w, img_h = size
    b = out_bbox.cpu().clone()
    b = box_cxcywh_to_xyxy(b)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def box_cxcywh_to_xyxy(x):
    """ Convert bounding boxes from [c_x, c_y, w, h] to [x_min, y_min, x_max, y_max] """
    x_c, y_c, w, h = x.unbind(1)
    b = [x_c - 0.5 * w, y_c - 0.5 * h, x_c + 0.5 * w, y_c + 0.5 * h]
    return torch.stack(b, dim=1)

def draw_box(frame, box, label):
    """ Draw a single bounding box on the frame """
    x_min, y_min, x_max, y_max = map(int, box)
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

# Open the video file
video_path = r'C:\\Users\\user\\Desktop\\Object-Detection-And-Tracking-Using-YOLOv8-And-TransTrack\\VID-20240720-WA0003.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video file at {video_path}.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("End of video or error reading frame.")
        break

    if tracker_initialized:
        # Update the tracker if it has been initialized
        success, tracked_bbox = tracker.update(frame)
        if success:
            x_min, y_min, w, h = map(int, tracked_bbox)
            x_max, y_max = x_min + w, y_min + h
            frame = draw_box(frame, (x_min, y_min, x_max, y_max), "bottle")
        else:
            # If tracking fails, reset and reinitialize the tracker in the next loop
            tracker_initialized = False
    else:
        if USE_DETR:
            # DETR processing
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img = transform(pil_img).unsqueeze(0)
            
            with torch.no_grad():
                outputs = model(img)
            
            boxes, labels, confidences = get_bounding_boxes_detr(outputs)
            boxes = rescale_bboxes(boxes, pil_img.size)

            if boxes.size(0) > 0:
                # Initialize the tracker with the first detected object
                object_bbox = boxes[0].numpy()
                x_min, y_min, x_max, y_max = map(int, object_bbox)
                
                # Initialize the tracker
                # Replace with actual TransTrack initialization
                # tracker = TranStrack()  # Uncomment if TransTrack is available
                tracker = cv2.TrackerMIL_create()  # Placeholder if TransTrack is not available
                tracker.init(frame, (x_min, y_min, x_max - x_min, y_max - y_min))
                tracker_initialized = True

        else:
            # YOLOv8 processing
            results = model(frame)
            annotated_frame = results[0].plot()
            
            if results[0].boxes:
                boxes = results[0].boxes.xyxy.numpy().astype(int)
                confidences = results[0].boxes.conf.numpy()
                labels = results[0].names  # Get the class names from YOLO

                if len(boxes) > 0:
                    # Initialize the tracker with the first detected object
                    object_bbox = boxes[0]
                    x_min, y_min, x_max, y_max = object_bbox
                    
                    # Initialize the tracker
                    # Replace with actual TransTrack initialization
                    # tracker = TranStrack()  # Uncomment if TransTrack is available
                    tracker = cv2.TrackerMIL_create()  # Placeholder if TransTrack is not available
                    tracker.init(frame, (x_min, y_min, x_max - x_min, y_max - y_min))
                    tracker_initialized = True

    # Display the frame
    cv2.imshow("Object Detection and Tracking", frame)
    
    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and destroy any OpenCV windows
cap.release()
cv2.destroyAllWindows()
