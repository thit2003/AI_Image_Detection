# 🧾 AI_Image_Detection

## Receipt Data Detection with YOLOv11

This project detects key information such as `seller_name`, `seller_id`, `total_items`, and `amount` from input receipt images using **YOLOv11** and deep learning.

---

## 📁 Project Structure

```
My Model V11/
├── main.py              # 🧠 Main program for image detection
├── best.pt              # 🏆 Trained YOLOv11 model
├── yolo11n.pt           # ⚙️ YOLOv11 base model used for training
├── output_images/       # 🖼️ Folder to save predicted images
├── model/               # 📊 Training outputs
│   ├── confusion_matrix.png
│   ├── P_curve.png
│   ├── results.csv
│   └── weights/
│       ├── best.pt
│       └── last.pt
├── DataSets/            # 📂 Dataset structure for training
│   ├── data.yaml
│   ├── train/
│   ├── valid/
│   └── test/
└── labelme/             # 🏷️ Labeling files created with LabelMe
```

---

## 🛠️ Tools & Technologies

- 🔍 **Object Detection**: [YOLOv11](https://github.com/WongKinYiu/yolov11)
- 🏷️ **Labeling Tool**: [LabelMe](https://github.com/wkentaro/labelme)
- 🖼️ **Input**: Receipt images
- 📤 **Output**: Bounding boxes with classified receipt fields
- 💻 **Training**: Google Colab (T4 GPU)
- ☁️ **Storage**: Google Drive

---

## 🚀 How to Use

1. **Label your data** using LabelMe and export as YOLO format.
2. **Organize your dataset** into `train`, `valid`, and `test` folders under `DataSets/`.
3. **Train your model** in Google Colab using T4 GPU and YOLOv11:
   - Use `yolo11n.pt` as the base weights.
4. **Run detection** by executing `main.py`.
5. **Check results**:
   - Predictions are saved in `output_images/`
   - Training performance is visualized in `model/` with confusion matrix, precision-recall curve, and results log.

---

## 🧠 Sample Output

Predicted receipts will display bounding boxes labeled with:
- Seller Name 🏪
- Seller ID 🆔
- Total Items 📦
- Amount 💵

---

## 📌 Notes

- Make sure your Google Drive is mounted properly in Colab.
- Adjust `data.yaml` to reflect your class names and path settings.
- Keep `main.py` updated to use `best.pt` for accurate inference.

---

## 📫 Contact

If you'd like to contribute or have questions, feel free to reach out or open an issue!

---
