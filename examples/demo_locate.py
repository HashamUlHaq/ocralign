from ocralign import process_image, locate_substring

page = process_image("./sample.png")

print("--- extracted text ---")
print(page.text)

print("\n--- overlay boxes for 'Simple Table' ---")
for occurrence in locate_substring(page, "Simple Table"):
    for box in occurrence:
        x0, y0, x1, y1 = box  # fractions (0-1) of page width/height
        print(f"x0={x0:.4f} y0={y0:.4f} x1={x1:.4f} y1={y1:.4f}")
