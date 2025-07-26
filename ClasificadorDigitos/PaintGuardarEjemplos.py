import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageDraw

class PaintApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Paint - Guardar Ejemplos 28x28")
        
        # Crear un lienzo para dibujar
        self.canvas = tk.Canvas(root, width=280, height=280, bg="white")
        self.canvas.pack()
        
        # Configurar eventos del mouse
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset_coords)
        
        # Botones para guardar y limpiar
        self.save_button = tk.Button(root, text="Guardar", command=self.save_image)
        self.save_button.pack(side=tk.LEFT, padx=10, pady=10)
        
        self.clear_button = tk.Button(root, text="Limpiar", command=self.clear_canvas)
        self.clear_button.pack(side=tk.RIGHT, padx=10, pady=10)
        
        # Variables para dibujar
        self.image = Image.new("L", (280, 280), "white")  # Imagen en escala de grises
        self.draw = ImageDraw.Draw(self.image)
        self.last_x, self.last_y = None, None

    def paint(self, event):
        x, y = event.x, event.y
        if self.last_x and self.last_y:
            # Dibujar en el lienzo y en la imagen
            self.canvas.create_line(self.last_x, self.last_y, x, y, fill="black", width=8)
            self.draw.line([self.last_x, self.last_y, x, y], fill="black", width=8)
        self.last_x, self.last_y = x, y

    def reset_coords(self, event):
        self.last_x, self.last_y = None, None

    def clear_canvas(self):
        # Limpiar el lienzo y la imagen
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, 280, 280], fill="white")

    def save_image(self):
        # Redimensionar la imagen a 28x28 y guardar como PNG
        file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
        if file_path:
            resized_image = self.image.resize((28, 28), Image.Resampling.LANCZOS)  
            resized_image.save(file_path)
            print(f"Imagen guardada en: {file_path}")

# Crear la ventana principal
if __name__ == "__main__":
    root = tk.Tk()
    app = PaintApp(root)
    root.mainloop()