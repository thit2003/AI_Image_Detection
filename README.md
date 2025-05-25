# AI_Image_Detection

🧾 Receipt Data Detection with YOLOv11
This project detects key information such as seller_name, seller_id, total_items, and amount from input receipt images using YOLOv11 and deep learning.

📂 Project Structure
My Model V11/
├── main.py                       # 🧠 Main program for image detection
├── best.pt                      # 🏆 Trained YOLOv11 model
├── yolo11n.pt                   # ⚙️ YOLOv11 base model used for training
├── output_images/               # 🖼️ Folder to save predicted images
├── model/                       # 📊 Training outputs
│   ├── confusion_matrix.png
│   ├── P_curve.png
│   ├── results.csv
│   └── weights/
│       ├── best.pt
│       └── last.pt
├── DataSets/                    # 📁 Dataset structure for training
│   ├── data.yaml
│   ├── train/
│   ├── valid/
│   └── test/
└── labelme/                     # 🏷️ Labeling files created with LabelMe

🛠️ Tools & Technologies
🔍 Object Detection: YOLOv11
🧰 Labeling Tool: LabelMe
🖼️ Input: Receipt images
📈 Output: Bounding boxes with classified data fields
🧪 Training: Google Colab (T4 GPU)
☁️ Storage: Google Drive

🚀 How It Works
📷 Upload receipt images
🏷️ Annotate using LabelMe
🏋️ Train the model on Colab using yolo11n.pt base
🔍 Use main.py to run detections
🖼️ Save predictions in output_images/
📊 Evaluate training results in model/

📝 Notes
Make sure to mount Google Drive in Google Colab to access "My Model V11" directory.
data.yaml is configured for YOLOv11 with custom class names related to receipt fields.
