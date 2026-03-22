import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np
import tensorflow as tf

# Load trained model
model = tf.keras.models.load_model("digit_model.h5")

# Create GUI
root = tk.Tk()
root.title("Handwritten Digit Recognition")

canvas = tk.Canvas(root, width=280, height=280, bg="white")
canvas.pack()

image = Image.new("L", (280, 280), 255)
draw = ImageDraw.Draw(image)

def paint(event):
    x1, y1 = event.x-8, event.y-8
    x2, y2 = event.x+8, event.y+8
    canvas.create_oval(x1,y1,x2,y2, fill="black")
    draw.ellipse([x1,y1,x2,y2], fill=0)

canvas.bind("<B1-Motion>", paint)

def predict_digit():
    img = image.resize((28,28))
    img = np.array(img)
    img = img / 255.0
    img = img.reshape(1,28,28,1)

    prediction = model.predict(img)
    digit = np.argmax(prediction)

    result_label.config(text=f"Predicted Digit: {digit}")

def clear_canvas():
    canvas.delete("all")
    draw.rectangle([0,0,280,280], fill=255)
    result_label.config(text="")

btn_predict = tk.Button(root, text="Predict", command=predict_digit)
btn_predict.pack()

btn_clear = tk.Button(root, text="Clear", command=clear_canvas)
btn_clear.pack()

result_label = tk.Label(root, text="", font=("Arial", 20))
result_label.pack()

root.mainloop()