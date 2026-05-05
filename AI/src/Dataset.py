import os
import random
from PIL import Image, ImageDraw, ImageFont

# 1. Setup the folder
folder_name = "data"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

def generate_algerian_plate(text, plate_type="yellow"):
    width, height = 520, 110
    bg_color = (255, 215, 0) if plate_type == "yellow" else (255, 255, 255)
    img = Image.new('RGB', (width, height), color=bg_color)
    d = ImageDraw.Draw(img)
    
    # 1. TRY DIFFERENT FONT PATHS (Linux/WSL standard paths)
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "arialbd.ttf", # Your local project folder
    ]
    
    selected_font = None
    font_size = 62  # High starting size

    for path in font_paths:
        if os.path.exists(path):
            try:
                selected_font = ImageFont.truetype(path, font_size)
                break
            except:
                continue

    if selected_font is None:
        print("Warning: No TTF fonts found. Using tiny default font.")
        selected_font = ImageFont.load_default()

    # 2. MEASURE AND CENTER
    bbox = d.textbbox((0, 0), text, font=selected_font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    x = (width - text_width) / 2
    y = (height - text_height) / 2 - 5 # Adjust vertical alignment

    d.text((x, y), text, fill=(0, 0, 0), font=selected_font)
    return img
# 4. Generation Loop
total_images = 300
for i in range(total_images):
    # Generating standard 5 or 6 digit serials
    serial = random.randint(10000, 999999)
    cat_year = f"1{random.randint(15, 26)}" # Years 2015 to 2026
    wilaya = f"{random.randint(1, 58):02d}"
    
    plate_text = f"{serial} {cat_year} {wilaya}"
    color = "yellow" if i % 2 == 0 else "white"
    
    plate_img = generate_algerian_plate(plate_text, plate_type=color)
    plate_img.save(os.path.join(folder_name, f"plate_{i:03d}.png"))

print(f"Done! Created 300 high-fill images in '{folder_name}'.")