
import os
import re
import mimetypes
import fitz
import pytesseract

from google.genai import types

from config import client, MODEL_NAME

from PIL import (
    Image,
    ImageEnhance,
    ImageFilter,
    ImageOps
)

from docx import Document
from docx.table import Table as DocxTable
from docx.text.paragraph import Paragraph


# ==========================================
# STEP 1: DOCUMENT TEXT EXTRACTION
# ==========================================

def extract_text(file_path):

    extension = os.path.splitext(file_path)[1].lower()

    if extension == ".pdf":
        return extract_pdf(file_path)

    elif extension == ".docx":
        return extract_docx(file_path)

    elif extension in [".png", ".jpg", ".jpeg"]:
        return extract_image(file_path)

    else:
        raise ValueError("Unsupported file format")


# ==========================================
# STEP 2: PDF EXTRACTION WITH TABLE SUPPORT
# ==========================================

def extract_pdf(file_path):

    extracted_pages = []

    with fitz.open(file_path) as pdf:

        for page_number, page in enumerate(pdf, start=1):

            page_content = []

            try:
                tables = page.find_tables()
                found_tables = tables.tables

            except Exception:
                found_tables = []

            text_blocks = page.get_text("blocks", sort=True)

            table_regions = [
                table.bbox for table in found_tables
            ]

            normal_text = []

            for block in text_blocks:

                if len(block) < 5:
                    continue

                x0, y0, x1, y1, text = block[:5]

                inside_table = False

                for bbox in table_regions:

                    tx0, ty0, tx1, ty1 = bbox

                    overlap_x = min(x1, tx1) - max(x0, tx0)
                    overlap_y = min(y1, ty1) - max(y0, ty0)

                    if overlap_x > 0 and overlap_y > 0:
                        inside_table = True
                        break

                if not inside_table and text.strip():
                    normal_text.append(text.strip())

            if normal_text:
                page_content.append("\n".join(normal_text))

            for table_number, table in enumerate(
                found_tables, start=1
            ):

                rows = table.extract()

                if not rows:
                    continue

                page_content.append(f"\nTABLE {table_number}:")

                for row in rows:

                    cleaned_cells = [
                        str(cell).strip().replace("\n", " ")
                        if cell is not None else ""
                        for cell in row
                    ]

                    row_text = " | ".join(cleaned_cells)

                    if row_text.strip(" |"):
                        page_content.append(row_text)

            extracted_pages.append(
                f"\n--- PAGE {page_number} ---\n"
                + "\n".join(page_content)
            )

    return "\n".join(extracted_pages).strip()


# ==========================================
# STEP 3: WORD EXTRACTION WITH TABLE SUPPORT
# ==========================================

def extract_docx(file_path):

    document = Document(file_path)

    content = []

    for element in document.element.body.iterchildren():

        if element.tag.endswith("}p"):

            paragraph = Paragraph(element, document)

            if paragraph.text.strip():
                content.append(paragraph.text.strip())

        elif element.tag.endswith("}tbl"):

            table = DocxTable(element, document)

            content.append("\nTABLE:")

            for row in table.rows:

                row_text = " | ".join(
                    cell.text.strip().replace("\n", " ")
                    for cell in row.cells
                )

                content.append(row_text)

    return "\n".join(content).strip()


# ==========================================
# STEP 4: GEMINI VISION IMAGE EXTRACTION
# ==========================================

def extract_image_with_gemini(file_path):

    print("\nUsing Gemini Vision for image extraction...")

    mime_type, _ = mimetypes.guess_type(file_path)

    if not mime_type:
        mime_type = "image/jpeg"

    with open(file_path, "rb") as image_file:
        image_bytes = image_file.read()

    prompt = """
You are an accurate document and image text extraction system.

Carefully examine the supplied image and extract all readable information.

IMPORTANT INSTRUCTIONS:

1. Extract headings, labels, numbers, descriptions and other readable text.
2. Preserve the original wording wherever possible.
3. Identify tables and reconstruct them in the correct row and column order.
4. Keep each table row's values together.
5. Do not mix the Objective and Key Outcome columns.
6. Associate a value with a mission or heading only when the image layout
   supports that association.
7. Preserve dates, percentages, names and numerical values accurately.
8. For dashboards, preserve section names and the values displayed under them.
9. Do not invent missing or unreadable text.
10. Mark unclear text as [Unclear] instead of guessing.

OUTPUT FORMAT:

DOCUMENT TEXT:

Extract normal headings and paragraphs in reading order.

TABLE 1:

Column 1 | Column 2 | Column 3

Row value | Row value | Row value

For tables with multiple columns, preserve every column separately.

Return only the extracted document content.
Do not answer questions or summarize the image.
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[
            types.Part.from_bytes(
                data=image_bytes,
                mime_type=mime_type
            ),
            prompt
        ]
    )

    extracted_text = response.text

    if not extracted_text or not extracted_text.strip():
        raise ValueError("Gemini Vision returned empty text")

    extracted_text = extracted_text.strip()

    print("\n========== GEMINI VISION RESULT ==========")
    print(extracted_text)
    print("==========================================")

    return extracted_text


# ==========================================
# STEP 5: CREATE MULTIPLE IMAGE VARIANTS
# ==========================================

def preprocess_image(image):

    image = image.convert("RGB")

    if image.width < 1800:

        scale = 1800 / image.width

        image = image.resize(
            (
                int(image.width * scale),
                int(image.height * scale)
            ),
            Image.Resampling.LANCZOS
        )

    grayscale = ImageOps.grayscale(image)

    enhanced = ImageOps.autocontrast(grayscale)

    enhanced = ImageEnhance.Contrast(enhanced).enhance(1.7)

    sharpened = enhanced.filter(ImageFilter.SHARPEN)

    threshold = sharpened.point(
        lambda pixel: 255 if pixel > 165 else 0
    )

    return [
        ("enhanced", enhanced),
        ("sharpened", sharpened),
        ("threshold", threshold)
    ]


# ==========================================
# STEP 6: OCR TEXT QUALITY SCORING
# ==========================================

def calculate_ocr_score(text, confidence):

    words = re.findall(r"[A-Za-z0-9]+", text)

    useful_words = [
        word for word in words
        if len(word) > 1
    ]

    line_count = len([
        line for line in text.splitlines()
        if line.strip()
    ])

    score = (
        confidence * 0.45
        + min(len(useful_words), 150) * 0.35
        + min(line_count, 40) * 0.5
    )

    return score


# ==========================================
# STEP 7: PRESERVE OCR LINE STRUCTURE
# ==========================================

def extract_ocr_with_layout(image, config):

    data = pytesseract.image_to_data(
        image,
        lang="eng",
        config=config,
        output_type=pytesseract.Output.DICT
    )

    lines = {}
    confidence_values = []

    for i, word in enumerate(data["text"]):

        word = word.strip()

        if not word:
            continue

        try:
            confidence = float(data["conf"][i])
        except (ValueError, TypeError):
            continue

        if confidence < 0:
            continue

        confidence_values.append(confidence)

        line_key = (
            data["block_num"][i],
            data["par_num"][i],
            data["line_num"][i]
        )

        if line_key not in lines:
            lines[line_key] = []

        lines[line_key].append(word)

    extracted_lines = []

    for words in lines.values():

        line = " ".join(words).strip()

        if line:
            extracted_lines.append(line)

    extracted_text = "\n".join(extracted_lines)

    average_confidence = (
        sum(confidence_values) / len(confidence_values)
        if confidence_values else 0
    )

    return extracted_text, average_confidence


# ==========================================
# STEP 8: IMAGE EXTRACTION WITH FALLBACK
# ==========================================

def extract_image(file_path):

    # First attempt: Gemini Vision
    try:

        extracted_text = extract_image_with_gemini(file_path)

        if extracted_text.strip():
            print("\nImage extraction completed using Gemini Vision.")
            return extracted_text

    except Exception as e:

        print("\nGemini Vision extraction failed:", e)
        print("Switching to Tesseract OCR...")


    # Second attempt: Existing Tesseract OCR
    with Image.open(file_path) as original_image:

        image_variants = preprocess_image(original_image)

        ocr_configs = [
            "--oem 3 --psm 3",
            "--oem 3 --psm 6",
            "--oem 3 --psm 11",
            "--oem 3 --psm 12"
        ]

        best_text = ""
        best_score = -1
        best_confidence = 0
        best_method = ""

        for variant_name, processed_image in image_variants:

            for config in ocr_configs:

                try:

                    extracted_text, confidence = (
                        extract_ocr_with_layout(
                            processed_image,
                            config
                        )
                    )

                    if not extracted_text.strip():
                        continue

                    score = calculate_ocr_score(
                        extracted_text,
                        confidence
                    )

                    print(
                        f"OCR test: {variant_name} | "
                        f"{config} | "
                        f"Confidence: {confidence:.2f} | "
                        f"Score: {score:.2f}"
                    )

                    if score > best_score:

                        best_text = extracted_text
                        best_score = score
                        best_confidence = confidence

                        best_method = (
                            f"{variant_name} | {config}"
                        )

                except Exception as e:

                    print("OCR configuration error:", e)

        if not best_text.strip():

            best_text = pytesseract.image_to_string(
                original_image,
                lang="eng",
                config="--oem 3 --psm 3"
            ).strip()

        print("\n========== IMAGE OCR RESULT ==========")
        print(best_text)
        print("\nOCR confidence:", round(best_confidence, 2))
        print("Selected OCR method:", best_method)
        print("======================================")

        return best_text.strip()
