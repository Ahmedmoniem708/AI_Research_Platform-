"""
Document Processing Pipeline for the AI Knowledge Platform.
Handles image preprocessing and OCR text extraction, as well as PDF text extraction
with section heading and page number detection.
"""

import cv2
import numpy as np
from paddleocr import PaddleOCR
from typing import List, Dict, Any, Tuple
import logging
import fitz  # PyMuPDF
import os

logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self, lang: str = 'en'):
        """
        Initialize the document processor with PaddleOCR.
        :param lang: Language for OCR, supports 'en', 'ch', 'fr', 'german', 'korean', 'japan', 'arabic'
        """
        # Initialize PaddleOCR with the specified language
        self.ocr = PaddleOCR(use_angle_cls=True, lang=lang, show_log=False)

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess the image for better OCR results.
        Steps: denoising, binarization.
        :param image: Input image as numpy array (BGR format from OpenCV)
        :return: Preprocessed image (grayscale, binary)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Denoising using Non-local Means Denoising
        denoised = cv2.fastNlMeansDenoising(gray, h=10)

        # Binarization using Otsu's thresholding
        _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return binary

    def extract_text_from_image(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Extract text from the preprocessed image using PaddleOCR.
        :param image: Preprocessed image (grayscale, binary)
        :return: List of dictionaries containing text and bounding box information
        """
        # PaddleOCR expects RGB image, but our binary is grayscale. Convert to RGB for consistency.
        # Note: PaddleOCR can work with grayscale, but we'll convert to 3-channel to avoid issues.
        if len(image.shape) == 2:  # Grayscale
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = image

        # Run OCR
        result = self.ocr.ocr(image_rgb, cls=True)

        # Process the result to extract text and bounding boxes
        extracted_data = []
        if result[0] is not None:
            for line in result[0]:
                # line format: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]], (text, confidence)
                bbox, (text, confidence) = line
                extracted_data.append({
                    "text": text,
                    "confidence": float(confidence),
                    "bbox": bbox  # List of 4 points
                })

        return extracted_data

    def _process_pdf(self, file_contents: bytes, filename: str) -> List[Dict[str, Any]]:
        """
        Process a PDF document: extract text, page numbers, and section headings.
        :param file_contents: The PDF file as bytes.
        :param filename: The name of the file (for metadata).
        :return: List of chunks, each containing text and metadata (page_number, heading_path).
        """
        chunks = []
        try:
            # Open the PDF from bytes
            pdf_document = fitz.open(stream=file_contents, filetype="pdf")

            # Get the outline (table of contents) for heading paths
            outline = pdf_document.get_toc()  # Returns list of [level, title, page_num]
            # We'll create a mapping from page number to heading path for quick lookup
            # For simplicity, we'll assume the outline is hierarchical and we can build a path for each heading.
            # We'll create a dict: heading_path_by_page[page_num] = list of heading strings from root to that page's section.
            # However, note: a page may contain multiple headings. We'll assign the heading path based on the last heading on or before the page.
            # For simplicity, we'll just use the outline to provide heading paths for each heading, and then for each text block we'll
            # assign the heading path of the most recent heading before that block.
            # But given the complexity, we'll do a simpler approach: for each page, we'll compute the heading path by looking at the outline
            # and finding the deepest heading that starts on or before the current page.

            # Build a list of outline entries with their page numbers (1-indexed in PyMuPDF)
            outline_entries = []
            for level, title, page_num in outline:
                outline_entries.append({
                    "level": level,
                    "title": title,
                    "page": page_num  # PyMuPDF page numbers are 1-indexed
                })

            # For each page in the PDF
            for page_num in range(len(pdf_document)):
                page = pdf_document.load_page(page_num)  # page_num is 0-indexed
                text = page.get_text("text")  # Get plain text

                # Determine the heading path for this page
                heading_path = []
                # Find the most recent outline entry that is at or before this page (page_num+1 because outline pages are 1-indexed)
                for entry in reversed(outline_entries):
                    if entry["page"] <= page_num + 1:
                        # We found a heading that starts at or before this page.
                        # Now we need to build the path from the root to this heading.
                        # We'll collect all outline entries that are ancestors of this entry.
                        path = []
                        current_level = entry["level"]
                        # We'll walk backwards in the outline to find parents.
                        # Since outline is hierarchical, we can look for entries with lower level that come before.
                        temp_path = [entry["title"]]
                        for ancestor in reversed(outline_entries[:outline_entries.index(entry)]):
                            if ancestor["level"] < current_level:
                                temp_path.append(ancestor["title"])
                                current_level = ancestor["level"]
                        heading_path = list(reversed(temp_path))
                        break

                # If no heading found, heading_path remains empty list.

                # Split the text into chunks (we can use simple chunking by sentences or fixed size)
                # For now, we'll chunk by paragraphs (double newline) and then if too big, split further.
                paragraphs = text.split('\n\n')
                for para in paragraphs:
                    if para.strip():
                        # Further split if paragraph is too long (e.g., > 500 characters)
                        if len(para) > 500:
                            # Split by sentences (simple split by '. ')
                            sentences = para.split('. ')
                            current_chunk = ""
                            for sent in sentences:
                                if len(current_chunk) + len(sent) < 500:
                                    current_chunk += sent + ". "
                                else:
                                    if current_chunk:
                                        chunks.append({
                                            "text": current_chunk.strip(),
                                            "metadata": {
                                                "doc_id": filename,
                                                "page_number": page_num + 1,
                                                "heading_path": heading_path.copy(),
                                                "citation_tag": f"{filename}, p.{page_num + 1}" + (f" - {' > '.join(heading_path)}" if heading_path else "")
                                            }
                                        })
                                    current_chunk = sent + ". "
                            if current_chunk:
                                citation_tag = f"{filename}, p.{page_num + 1}" + (f" - {' > '.join(heading_path)}" if heading_path else "")
                                chunks.append({
                                    "text": current_chunk.strip(),
                                    "metadata": {
                                        "doc_id": filename,
                                        "page_number": page_num + 1,
                                        "heading_path": heading_path.copy(),
                                        "citation_tag": citation_tag
                                    }
                                })
                        else:
                            chunks.append({
                                "text": para.strip(),
                                "metadata": {
                                    "doc_id": filename,
                                    "page_number": page_num + 1,
                                    "heading_path": heading_path.copy(),
                                    "citation_tag": f"{filename}, p.{page_num + 1}" + (f" - {' > '.join(heading_path)}" if heading_path else "")
                                }
                            })

            pdf_document.close()
            logger.info(f"Processed PDF '{filename}' with {len(chunks)} chunks.")

        except Exception as e:
            logger.error(f"Error processing PDF {filename}: {str(e)}")
            # Fallback to treating as image? Or just return empty?
            # For now, return empty list and let the caller handle.
            return []

        return chunks

    def process_document(self, file_contents: bytes, filename: str) -> List[Dict[str, Any]]:
        """
        Process a document (image or PDF) and extract text with metadata.
        :param file_contents: The file contents as bytes.
        :param filename: The name of the file (used to determine type and for metadata).
        :return: List of chunks, each containing text and metadata.
        """
        # Determine file type by extension
        file_extension = os.path.splitext(filename)[1].lower()

        if file_extension == '.pdf':
            return self._process_pdf(file_contents, filename)
        else:
            # Treat as image (assuming common image formats)
            try:
                # Convert bytes to OpenCV image
                nparr = np.frombuffer(file_contents, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if image is None:
                    logger.error(f"Could not decode image file {filename}")
                    return []

                # Preprocess and extract text
                preprocessed_image = self.preprocess_image(image)
                extracted_text_data = self.extract_text_from_image(preprocessed_image)

                # Convert extracted text data to chunks format
                chunks = []
                for i, text_block in enumerate(extracted_text_data):
                    citation_tag = f"{filename}, p.1"  # Single page for images
                    chunks.append({
                        "text": text_block["text"],
                        "metadata": {
                            "doc_id": filename,
                            "page_number": 1,  # Assume single page for images
                            "heading_path": [],  # No heading structure for images
                            "citation_tag": citation_tag
                        }
                    })

                logger.info(f"Processed image '{filename}' with {len(chunks)} text blocks.")
                return chunks

            except Exception as e:
                logger.error(f"Error processing image {filename}: {str(e)}")
                return []

# Example usage (for testing)
if __name__ == "__main__":
    # This is just for demonstration; in practice, you would load an image or PDF from a file or upload.
    processor = DocumentProcessor(lang='en')  # Change to 'arabic' for Arabic support
    # Assume we have an image loaded as a numpy array (e.g., from cv2.imread)
    # image = cv2.imread("sample.jpg")
    # preprocessed, text_data = processor.process_document(image)
    # print(f"Extracted {len(text_data)} text blocks")
    pass