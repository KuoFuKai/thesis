import multiprocessing
from ultralytics import YOLO
import torch
from ultralytics.data.utils import autosplit

if __name__ == '__main__':
    # Auto split
    # autosplit(path='C:/Users/kevin/PycharmProjects/Datasets/Tainan_Confucian_Temple/images', weights=(0.8, 0.1, 0.1), )

    multiprocessing.freeze_support()
    # Set GPU
    torch.cuda.set_device(0)

    # Load a model
    model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

    # Train the model
    results = model.train(data='C:/Users/kevin/PycharmProjects/Datasets/Tainan_Confucian_Temple/data.yaml', epochs=500)

    # Evaluate
    metrics = model.val()  # evaluate model performance on the validation set