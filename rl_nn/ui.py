import tkinter as tk
from tkinter import filedialog, Label
from tkinterdnd2 import DND_FILES, TkinterDnD
from PIL import Image, ImageTk
import numpy as np
from nn import rlNeuralNetwork

# Race label list
race_labels = ["Azael", "Morvid", "Navaran", "Scroom"]

# Neural network parameters
input_size = 64 * 64 * 3
net = rlNeuralNetwork(input_size=input_size, hidden_size=64, output_size=4, lr=0.1)

# Load trained weights
weights = np.load("model_weights.npz")
net.W1 = weights["W1"]
net.b1 = weights["b1"]
net.W2 = weights["W2"]
net.b2 = weights["b2"]

# Preprocessing function
def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB").resize((64, 64))
    img_array = np.asarray(img) / 255.0
    return img_array.reshape(-1, 1)  # Flatten to (12288, 1)

# Classification function
def classify_image(path):
    x = preprocess_image(path)
    output, _ = net.forward_propagation(x)
    probs = output.flatten()
    pred_index = np.argmax(probs)
    return race_labels[pred_index], probs

# Image handling and UI update
def handle_image(path):
    try:
        img = Image.open(path).resize((160, 160))
        img_tk = ImageTk.PhotoImage(img)
        image_label.config(image=img_tk)
        image_label.image = img_tk

        race, probs = classify_image(path)
        result_label.config(text=f"Predicted Race: {race}", fg="cyan")

        prob_lines = [f"{race_labels[i]}: {probs[i]*100:.2f}%" for i in range(len(race_labels))]
        prob_label.config(text="\n".join(prob_lines))

    except Exception as e:
        result_label.config(text="Invalid image file.", fg="red")
        prob_label.config(text="")
        image_label.config(image="")

# Drag-and-drop handler
def drop(event):
    path = event.data.strip('{}')  # Handles file paths with spaces
    handle_image(path)

# File picker handler
def open_file_dialog():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg")])
    if file_path:
        handle_image(file_path)

# Setup GUI window
window = TkinterDnD.Tk()
window.title("Rogue Lineage Race Classifier")
window.geometry("400x550")
window.configure(bg="#1e1e1e")

# Title
title_label = Label(window, text="Drag & Drop an Image", font=("Helvetica", 18),
                    bg="#1e1e1e", fg="white")
title_label.pack(pady=10)

# Drop area
drop_area = Label(window, text="Drop image here", bg="#333", fg="white",
                  relief="groove", width=35, height=6, font=("Helvetica", 12))
drop_area.pack(pady=10)
drop_area.drop_target_register(DND_FILES)
drop_area.dnd_bind("<<Drop>>", drop)

# Choose file button
choose_btn = tk.Button(window, text="Choose Image File", command=open_file_dialog,
                       bg="#555", fg="white", font=("Helvetica", 12))
choose_btn.pack(pady=10)

# Image preview
image_label = Label(window, bg="#1e1e1e")
image_label.pack(pady=10)

# Prediction result
result_label = Label(window, text="", font=("Helvetica", 14),
                     bg="#1e1e1e", fg="cyan")
result_label.pack(pady=5)

# Softmax probabilities
prob_label = Label(window, text="", font=("Helvetica", 12),
                   bg="#1e1e1e", fg="white", justify="left")
prob_label.pack(pady=5)

# Start the app
window.mainloop()

