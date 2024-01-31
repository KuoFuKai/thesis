import multiprocessing
from ultralytics import YOLO
import torch

if __name__ == '__main__':
    multiprocessing.freeze_support()
    # Set GPU
    torch.cuda.set_device(0)

    # Load a model
    model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

    # Train the model
    results = model.train(data='C:/Users/kevin/PycharmProjects/Datasets/My_first_project/data.yaml', epochs=100)

    # Evaluate
    metrics = model.val()  # evaluate model performance on the validation set