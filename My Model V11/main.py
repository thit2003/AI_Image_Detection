import os
import json
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import torch
from ultralytics import YOLO

# Load YOLO model
model = YOLO("best.pt")

# Define output folder
output_folder = r"C:\Users\Thit Lwin Win Thant\Downloads\Project\My Model V11\output_images"
os.makedirs(output_folder, exist_ok=True)  # Ensure output folder exists

def select_image():
    """Open file dialog to select an image and process it."""
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png;*.jpeg")])
    if file_path:
        process_image(file_path)

def process_image(file_path):
    """Run YOLO on the selected image and save the output."""
    image = cv2.imread(file_path)
    results = model(image, conf=0.25)  # Lower confidence threshold to 0.25
    detections = []
    
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            label = result.names[int(box.cls[0])]
            
            # Append detection data
            detections.append({
                "label": label,
                "confidence": round(conf, 2),
                "bbox": [x1, y1, x2, y2]
            })
            
            # Draw bounding box and label
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Debugging: Print detections
    print("Detections:", detections)
    
    save_processed_image(image, file_path)
    save_json_results(file_path, detections)
    display_image(image)

def save_processed_image(image, original_path):
    """Save processed image with bounding boxes to output folder."""
    file_name = os.path.basename(original_path)
    output_path = os.path.join(output_folder, file_name)
    cv2.imwrite(output_path, image)

def save_json_results(original_path, detections):
    """Save detection results as a JSON file in LabelMe format."""
    file_name = os.path.splitext(os.path.basename(original_path))[0] + ".json"
    json_path = os.path.join(output_folder, file_name)
    
    labelme_format = {
        "version": "4.5.9",  # LabelMe version
        "imagePath": os.path.basename(original_path),
        "imageData": None,
        "shapes": []
    }
    
    for det in detections:
        shape = {
            "label": det["label"],
            "points": [
                [det["bbox"][0], det["bbox"][1]],  # Top-left (x1, y1)
                [det["bbox"][2], det["bbox"][3]]   # Bottom-right (x2, y2)
            ],
            "group_id": None,
            "shape_type": "rectangle",
            "flags": {}
        }
        labelme_format["shapes"].append(shape)

    with open(json_path, "w") as json_file:
        json.dump(labelme_format, json_file, indent=4)
    
    # Debugging: Print saved JSON path
    print(f"JSON saved at: {json_path}")

def display_image(image):
    """Display the processed image in the GUI."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image = image.resize((500, 400), Image.LANCZOS)
    photo = ImageTk.PhotoImage(image)
    panel.config(image=photo)
    panel.image = photo

# GUI Setup
root = tk.Tk()
root.title("Object Detection")
root.geometry("600x500")
root.configure(bg="#1E1E1E")

frame = tk.Frame(root, bg="#2D2D2D", padx=10, pady=10)
frame.pack(pady=20)

panel = tk.Label(frame, bg="#2D2D2D")
panel.pack()

btn_select = tk.Button(root, text="Select Image", command=select_image,
                        font=("Arial", 14), bg="#007ACC", fg="white", 
                        padx=10, pady=5, relief="flat")
btn_select.pack(pady=20)

root.mainloop()