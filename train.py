from ultralytics  import YOLO



if __name__ == "__main__":
    model = YOLO('yolo11n.yaml')
    model.train(
        data=r'F:\Project_py\datasets\african-wildlife\african-wildlife.yaml',
        model='yolo11n.yaml',
        epochs=10,
        batch=8,
        device='0',
        imgsz=640,
        afss=True,
        workers=0,
        afss_interval=1,
    )
