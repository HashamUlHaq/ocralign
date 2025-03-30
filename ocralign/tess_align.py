import numpy as np
from bs4 import BeautifulSoup
import pytesseract

def parse_bbox(title):
    """
    Extracts the bounding box from a title string.
    Expected format: "bbox x0 y0 x1 y1; baseline ..." 
    Returns a tuple: (x0, y0, x1, y1)
    """
    title = title.strip()
    if title.startswith("bbox"):
        bbox = [ int(i) for i in title.split(";")[0].replace("bbox", "").strip().split(" ")]
        return bbox
    else:
        raise Exception ("Error parsing bbox:", title)

def vertical_overlap_ratio(bbox1, bbox2):
    """
    Computes vertical overlap ratio between two bboxes.
    Returns overlap ratio relative to the smaller box height.
    """
    y0_1, y1_1 = bbox1[1], bbox1[3]
    y0_2, y1_2 = bbox2[1], bbox2[3]
    overlap = max(0, min(y1_1, y1_2) - max(y0_1, y0_2))
    height1 = y1_1 - y0_1
    height2 = y1_2 - y0_2
    return overlap / min(height1, height2) if min(height1, height2) else 0

def process_page(page):
    row_data = [] 

    # Get hOCR output from Tesseract
    hocr = pytesseract.image_to_pdf_or_hocr(page, extension='hocr', config='--psm 12')

    # Parse hOCR using BeautifulSoup
    soup = BeautifulSoup(hocr, 'html.parser')
    line_spans = soup.find_all("span", class_="ocr_line")

    # Extract each span's bbox and text
    spans = []
    for span in line_spans:
        title = span.get("title", "")
        bbox = parse_bbox(title)
        text = span.get_text().replace("\n", " ").strip()
        if bbox and text:
            spans.append({"bbox": bbox, "text": text})

    # Compute global average character width from all spans
    char_widths = []
    for span in spans:
        bbox = span["bbox"]
        span_width = bbox[2] - bbox[0]
        text_length = len(span["text"])
        if text_length > 0:
            char_widths.append(span_width / text_length)
    global_avg_char_width = np.mean(char_widths) if char_widths else 7

    # Group spans into rows using 80% vertical overlap
    grouped_rows = []
    used = [False] * len(spans)
    for i in range(len(spans)):
        if used[i]:
            continue
        current_row = [spans[i]]
        used[i] = True
        for j in range(i+1, len(spans)):
            if used[j]:
                continue
            if vertical_overlap_ratio(spans[i]["bbox"], spans[j]["bbox"]) >= 0.8:
                current_row.append(spans[j])
                used[j] = True
        grouped_rows.append(current_row)

    # For each grouped row, compute the absolute horizontal and vertical positions
    for row in grouped_rows:
        # Sort the row by x-coordinate (left-to-right)
        row.sort(key=lambda span: span["bbox"][0])
        
        # For this row, the leftmost x coordinate determines its horizontal placement.
        row_left = min(span["bbox"][0] for span in row)
        # Compute the indentation (in characters) from the page's left edge.
        indent_spaces = int(round(row_left / global_avg_char_width))
        
        # Build the aligned line by processing each cell.
        aligned_line = " " * indent_spaces
        # For each cell, compute its desired starting column from its absolute x-coordinate.
        for span in row:
            cell_start = int(round(span["bbox"][0] / global_avg_char_width))
            # Ensure the current line is padded to the cell's column start.
            current_length = len(aligned_line)
            if current_length < cell_start:
                aligned_line += " " * (cell_start - current_length)
            aligned_line += span["text"] + " "  # append the text and a space
        
        aligned_line = aligned_line.rstrip()  # remove trailing spaces
        
        # Vertical boundaries for the row (in pixels)
        row_top = min(span["bbox"][1] for span in row)
        row_bottom = max(span["bbox"][3] for span in row)
        row_height = row_bottom - row_top if row_bottom > row_top else 1
        
        # Store the row's data
        row_data.append({
            "aligned_line": aligned_line,
            "row_top": row_top,
            "row_bottom": row_bottom,
            "row_height": row_height
        })

    # Sort rows top-to-bottom based on their y-coordinate
    row_data.sort(key=lambda r: r["row_top"])

    # Compute a global average line height (in pixels) from the grouped rows
    line_heights = [row["row_height"] for row in row_data]
    global_avg_line_height = np.mean(line_heights) if line_heights else 15

    # Build the final output string using absolute vertical positions.
    final_output = ""
    prev_row_bottom = 0
    for row in row_data:
        # Determine the vertical gap (in pixels) between the current row and the previous one.
        gap_pixels = row["row_top"] - prev_row_bottom
        # Convert the gap to number of newlines (using the global average line height).
        gap_newlines = int(round(gap_pixels / global_avg_line_height)) + 1
        if gap_newlines < 1:
            gap_newlines = 1
        final_output += "\n" * gap_newlines
        final_output += row["aligned_line"] #+ "\n"
        prev_row_bottom = row["row_bottom"]

    return final_output