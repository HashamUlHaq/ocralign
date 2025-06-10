# 🧾 ocralign

`ocralign` is an OCR utility built on top of Tesseract that preserves the layout and formatting of scanned documents. It supports both PDFs and images and outputs clean, structured text.

---

## 🔧 System Requirements

Before installing the Python package, you need to install some system dependencies required by `pytesseract` and `pdf2image`:

```bash
sudo apt update
sudo apt install -y tesseract-ocr
sudo apt install -y poppler-utils
```

## Installation
```pip install ocralign```

## Usage example
```
from ocralign import process_pdf, process_image

# OCR a single image
print(process_image("./sample.png"))

# OCR a multi-page PDF (returns list of text per page)
texts = process_pdf("./images-pdf.pdf", dpi=300)

# OCR a PDF and write result to a file
process_pdf("./images-pdf.pdf", dpi=300, output_path="test.txt")
```
### Input image:

![Sample OCR Input](./examples/sample.png)

### Extracted Text [📎 See full output here](./examples/output.txt)

```
Sample Tables                                                                                = Print

 Tables used in papers can be so simple that they are "informal" enough to be a sentence member and not
 require a caption, or they can be complex enough that they require spreadsheets spanning several pages.
 A table’s fundamental purpose should always be to visually simplify complex material, in particular when
 the table is designed to help the reader identify trends. Here, a simple table and a complex table are used
 to demonstrate how tables help writers to record and "visualize" information and data.


 Simple Table

 The simple table that follows, from a student's progress report to his advisor, represents how tables need
 not always be about data presentation. Here the rows and columns simply make it easy for the writer to
 present the necessary information with efficiency. This unnumbered and informal table, in effect, explains
 itself.




                     Plan for Weekly Progress for the Remainder of the Semester

      Week of     Contact Dr. Berinni for relevant literature suggestions.
      11/28       Read lit reviews from Vibrational Spectroscopy.
                  Research experimental methods used to test polyurethanes, including infrared (IR)
                  spectroscopy and nuclear magnetic resonance (NMR).

      Week of     Define specific ways that polyurethanes can be improved.
      12/5        Develop experimental plan.

      Week of     Create visual aids, depicting chemical reactions and experimental setups.
      12/12       Prepare draft of analytical report.

      Week of     Turn in copy of preliminary analytical report, to be expanded upon next semester.
      12/18





 Complex Table

 The following sample table is excerpted from a student's senior thesis about tests conducted on
 Pennsylvania coal. Note the specificity of the table’s caption. Also note the level of discussion following the
 table, and how the writer uses the data from the table to move toward an explanation of the trends that
 the table reveals.
```
