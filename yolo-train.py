from ultralytics import YOLO
import torch
import multiprocessing

if __name__ == '__main__':
    multiprocessing.freeze_support()
    print(torch.cuda.is_available())
    torch.cuda.set_device(0)  # Set to your desired GPU number

    model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

    # Use the model
    model.train(data="C:/Users/kevin/PycharmProjects/Datasets/My_first_project/data.yaml", epochs=10)  # train the model
    metrics = model.val()  # evaluate model performance on the validation set
