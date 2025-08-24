from PIL import Image, ImageDraw, ImageFont

# === Eingabedateien ===
img_stage1 = "3118138f-b6f3-475a-a0e4-c69b637a90f9.png"   # Environment
img_stage2 = "full_compare1.png"                          # Training
img_stage3 = "gradio2.png"                                # Comparison

# === Ausgabe ===
output_file = "preview_collage_square.png"

# Bilder laden
images = [Image.open(f) for f in [img_stage1, img_stage2, img_stage3]]

# Normgröße (alle Bilder auf gleiche Breite)
size = 600
resized = [im.resize((size, int(im.height * size / im.width))) for im in images]

# Canvas für Quadrat vorbereiten
collage = Image.new("RGB", (size * 2, size * 2), color="white")

# Schrift
try:
    font = ImageFont.truetype("DejaVuSans-Bold.ttf", 28)
except:
    font = ImageFont.load_default()

draw = ImageDraw.Draw(collage)

# Stage 1 links oben
collage.paste(resized[0], (0, 40))
draw.text((10, 10), "Stage 1: Environment ✅", fill="black", font=font)

# Stage 2 rechts oben
collage.paste(resized[1], (size, 40))
draw.text((size + 10, 10), "Stage 2: Training ⚙️", fill="black", font=font)

# Stage 3 unten mittig
x_offset = (size - resized[2].width // 2)
y_offset = size + 40
collage.paste(resized[2], (x_offset, y_offset))
draw.text((size // 2 - 80, size + 10), "Stage 3: Comparison 🔍", fill="black", font=font)

# Auf 1:1 Quadratform croppen oder verkleinern
final = collage.crop((0, 0, size * 2, size * 2))

# Speichern
final.save(output_file)
print(f"✅ Quadratische Collage gespeichert: {output_file}")
