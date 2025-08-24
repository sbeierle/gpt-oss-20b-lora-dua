from PIL import Image, ImageDraw, ImageFont

# === Eingabedateien anpassen ===
img_stage1 = "b200_Cuda12.png"   # Environment (Sweet Spot)
img_stage2 = "full_compare1.png"                          # Training / Terminal Compare
img_stage3 = "gradio2.png"                                # Gradio UI

# === Ausgabe ===
output_file = "preview_collage.png"

# === Bilder laden ===
images = [Image.open(f) for f in [img_stage1, img_stage2, img_stage3]]

# Bilder auf gleiche Höhe skalieren
min_height = min(im.height for im in images)
resized = [im.resize((int(im.width * min_height / im.height), min_height)) for im in images]

# Breite der Collage berechnen
total_width = sum(im.width for im in resized)

# Collage erstellen (weißer Hintergrund)
collage = Image.new("RGB", (total_width, min_height + 50), color="white")

# Schrift vorbereiten
try:
    font = ImageFont.truetype("DejaVuSans-Bold.ttf", 24)
except:
    font = ImageFont.load_default()

draw = ImageDraw.Draw(collage)

# Labels
labels = ["Stage 1: Environment ✅", "Stage 2: Training ⚙️", "Stage 3: Comparison 🔍"]

# Bilder platzieren + Label schreiben
x_offset = 0
for im, label in zip(resized, labels):
    collage.paste(im, (x_offset, 50))
    draw.text((x_offset + 10, 10), label, fill="black", font=font)
    x_offset += im.width

# Speichern
collage.save(output_file)
print(f"✅ Collage gespeichert: {output_file}")
