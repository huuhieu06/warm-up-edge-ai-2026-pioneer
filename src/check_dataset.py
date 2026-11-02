
from pathlib import Path

# Lấy thư mục gốc của project
ROOT = Path(__file__).resolve().parent.parent

train_images = ROOT / "data" / "train" / "images"
train_labels = ROOT / "data" / "train" / "labels"
test_dir = ROOT / "data" / "test"
classes_file = ROOT / "data" / "classes.txt"

print("=== KIỂM TRA DATASET ===")
print("Thư mục project:", ROOT)

# Kiểm tra các thư mục
for path in [train_images, train_labels, test_dir]:
    print(f"\n{path}:")
    print("Tồn tại:", path.exists())

    if path.exists():
        files = list(path.iterdir())
        print("Số mục:", len(files))
        print("5 mục đầu:", [f.name for f in files[:5]])

# Kiểm tra danh sách lớp
if classes_file.exists():
    print("\n=== DANH SÁCH CLASSES ===")
    print(classes_file.read_text(encoding="utf-8"))
else:
    print("\nKhông tìm thấy classes.txt")
