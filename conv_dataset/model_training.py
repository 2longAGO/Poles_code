from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n-seg.pt')  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data='poles.yaml', epochs=300, imgsz=640, workers=0) #batch=-1
'''
results = model.val()  # evaluate model performance on the validation set
results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
success = YOLO("yolov8m-pole_seg.pt").export(format="onnx")  # export a model to ONNX format
'''
# C:\Users\leefl\Desktop\Poles_code\datasets\coco_converted\images\val

# C:\Users\leefl\Desktop\Poles_code\conv_dataset\datasets\coco_converted\images\val