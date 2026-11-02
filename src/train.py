from ultralytics import YOLO

def main():
    # Tải mô hình YOLO11n đã được huấn luyện trước
    model = YOLO("yolo11n.pt")

    # Huấn luyện mô hình trên dataset của em
    model.train(
        data="D:/EdgeAI/warm-up/data/data.yaml",
        epochs=50,
        imgsz=640,
        batch=8,
        patience=10,
        project="D:/EdgeAI/warm-up/runs",
        name="my_model"
    )

    print("Huấn luyện hoàn tất!")

if __name__ == "__main__":
    main()