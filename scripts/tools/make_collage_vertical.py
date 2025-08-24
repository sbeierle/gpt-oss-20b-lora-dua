from PIL import Image, ImageDraw, ImageFont

# === Eingabedateien ===
img_stage1 = "b200_Cuda12.png"   # Environment (Sweet Spot)
img_stage2 = "full_compare1.png"                          # Training / Terminal Compare
img_stage3 = "gradio2.png"                                # Gradio UI

# === Ausgabe ===
output_file = "preview_collage_vertical.png"

# Bilder laden
images = [Image.open(f) for f in [img_stage1, img_stage2, img_stage3]]

# Auf gleiche Breite skalieren
min_width = min(im.width for im in images)
resized = [im.resize((min_width, int(im.height * min_width / im.width))) for im in images]

# Gesamthöhe berechnen
total_height = sum(im.height for im in resized)

# Collage erstellen (weißer Hintergrund)
collage = Image.new("RGB", (min_width, total_height + 150), color="white")

# Schrift vorbereiten
try:
    font = ImageFont.truetype("DejaVuSans-Bold.ttf", 28)
except:
    font = ImageFont.load_default()

draw = ImageDraw.Draw(collage)

# Labels
labels = ["Stage 1: Environment ✅", "Stage 2: Training ⚙️", "Stage 3: Comparison 🔍"]

# Bilder platzieren + Label drüber
y_offset = 0
for im, label in zip(resized, labels):
    draw.text((10, y_offset + 10), label, fill="black", font=font)
    collage.paste(im, (0, y_offset + 40))
    y_offset += im.height + 50

# Speichern
collage.save(output_file)
print(f"✅ Vertikale Collage gespeichert: {output_file}")
