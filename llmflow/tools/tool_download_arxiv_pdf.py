"""
PDF Tool for LLM Agents
=======================

A comprehensive Python tool for working with PDF files including reading,
editing, merging, splitting, and more. Designed for LLM agent frameworks.

Requirements:
    - pypdf (install with: pip install pypdf)
    - pdfplumber (install with: pip install pdfplumber)
    - reportlab (install with: pip install reportlab)
    - Pillow (install with: pip install Pillow)
    - pytesseract (install with: pip install pytesseract)
    - requests (install with: pip install requests)
    - Python 3.7+

Optional:
    - tesseract-ocr system package for OCR functionality

Example Usage:
    >>> pdf_tool = PDFTool()
    >>> text = pdf_tool.extract_text("document.pdf")
    >>> pdf_tool.merge_pdfs(["doc1.pdf", "doc2.pdf"], "merged.pdf")
    >>> pdf_tool.split_pdf("large.pdf", start_page=1, end_page=10)

Author: AI Assistant
Version: 1.0.0
"""

import os
import io
import json
import logging
import hashlib
import requests
import subprocess
import sys
import glob
from typing import Dict, Optional, Union, List, Any, Tuple, BinaryIO
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import tempfile
from datetime import datetime

try:
    from .tool_decorator import register_tool
except ImportError:
    # If tool_decorator doesn't exist, create a dummy decorator
    def register_tool(tags=None):
        def decorator(func):
            return func
        return decorator

# Check and install required packages
packages = {
    'pypdf': 'pypdf',
    'pdfplumber': 'pdfplumber',
    'reportlab': 'reportlab',
    'PIL': 'Pillow',
    'pytesseract': 'pytesseract'
}

for import_name, install_name in packages.items():
    try:
        if import_name == 'PIL':
            from PIL import Image, ImageDraw, ImageFont
        else:
            __import__(import_name)
    except ImportError:
        print(f"{install_name} is not installed. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", install_name])

# Import after installation
import pypdf
import pdfplumber
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
try:
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("Warning: pytesseract not available. OCR features will be disabled.")


class PDFOperation(Enum):
    """Types of PDF operations."""
    EXTRACT_TEXT = "extract_text"
    EXTRACT_IMAGES = "extract_images"
    MERGE = "merge"
    SPLIT = "split"
    ROTATE = "rotate"
    WATERMARK = "watermark"
    ENCRYPT = "encrypt"
    DECRYPT = "decrypt"
    COMPRESS = "compress"
    OCR = "ocr"
    ADD_PAGE = "add_page"
    DELETE_PAGE = "delete_page"
    EXTRACT_METADATA = "extract_metadata"
    UPDATE_METADATA = "update_metadata"
    CONVERT = "convert"
    FILL_FORM = "fill_form"


class ImageFormat(Enum):
    """Image formats for extraction."""
    PNG = "png"
    JPEG = "jpeg"
    TIFF = "tiff"
    BMP = "bmp"


@dataclass
class PDFInfo:
    """PDF document information."""
    filename: str
    num_pages: int
    file_size: int
    title: Optional[str] = None
    author: Optional[str] = None
    subject: Optional[str] = None
    creator: Optional[str] = None
    producer: Optional[str] = None
    creation_date: Optional[datetime] = None
    modification_date: Optional[datetime] = None
    encrypted: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PDFOperationResult:
    """Result of a PDF operation."""
    success: bool
    operation: PDFOperation
    output_file: Optional[str] = None
    message: Optional[str] = None
    data: Optional[Any] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class PDFTool:
    """
    A comprehensive PDF tool for LLM agents.
    
    This class provides various PDF manipulation capabilities including
    reading, editing, merging, splitting, and more.
    
    Attributes:
        logger (logging.Logger): Logger instance for this class
        temp_dir (str): Temporary directory for intermediate files
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the PDF tool.
        
        Args:
            logger: Logger instance (creates default if None)
        """
        self.logger = logger or self._setup_logger()
        self.temp_dir = tempfile.mkdtemp()
    
    def _setup_logger(self) -> logging.Logger:
        """Set up default logger configuration."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def get_pdf_info(self, pdf_path: str) -> Optional[PDFInfo]:
        """
        Get information about a PDF file.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            PDFInfo object or None if error
        """
        try:
            # Get file size
            file_size = os.path.getsize(pdf_path)
            
            # Open PDF
            with open(pdf_path, 'rb') as file:
                reader = pypdf.PdfReader(file)
                
                # Get metadata
                metadata = reader.metadata if reader.metadata else {}
                
                info = PDFInfo(
                    filename=os.path.basename(pdf_path),
                    num_pages=len(reader.pages),
                    file_size=file_size,
                    title=metadata.get('/Title'),
                    author=metadata.get('/Author'),
                    subject=metadata.get('/Subject'),
                    creator=metadata.get('/Creator'),
                    producer=metadata.get('/Producer'),
                    creation_date=metadata.get('/CreationDate'),
                    modification_date=metadata.get('/ModDate'),
                    encrypted=reader.is_encrypted,
                    metadata=dict(metadata)
                )
                
                return info
                
        except Exception as e:
            self.logger.error(f"Error getting PDF info: {str(e)}")
            return None
    
    def extract_text(
        self,
        pdf_path: str,
        start_page: Optional[int] = None,
        end_page: Optional[int] = None,
        use_pdfplumber: bool = True
    ) -> PDFOperationResult:
        """
        Extract text from PDF.
        
        Args:
            pdf_path: Path to PDF file
            start_page: Starting page (1-indexed)
            end_page: Ending page (inclusive)
            use_pdfplumber: Use pdfplumber for better extraction
            
        Returns:
            PDFOperationResult with extracted text
        """
        try:
            extracted_text = []
            
            if use_pdfplumber:
                # Use pdfplumber for better text extraction
                with pdfplumber.open(pdf_path) as pdf:
                    start_idx = (start_page - 1) if start_page else 0
                    end_idx = end_page if end_page else len(pdf.pages)
                    
                    for i in range(start_idx, min(end_idx, len(pdf.pages))):
                        page = pdf.pages[i]
                        text = page.extract_text()
                        if text:
                            extracted_text.append(f"=== Page {i+1} ===\n{text}")
            else:
                # Use pypdf
                with open(pdf_path, 'rb') as file:
                    reader = pypdf.PdfReader(file)
                    
                    start_idx = (start_page - 1) if start_page else 0
                    end_idx = end_page if end_page else len(reader.pages)
                    
                    for i in range(start_idx, min(end_idx, len(reader.pages))):
                        page = reader.pages[i]
                        text = page.extract_text()
                        extracted_text.append(f"=== Page {i+1} ===\n{text}")
            
            full_text = "\n\n".join(extracted_text)
            
            return PDFOperationResult(
                success=True,
                operation=PDFOperation.EXTRACT_TEXT,
                data=full_text,
                message=f"Extracted text from {len(extracted_text)} pages",
                metadata={'num_pages': len(extracted_text)}
            )
            
        except Exception as e:
            self.logger.error(f"Error extracting text: {str(e)}")
            return PDFOperationResult(
                success=False,
                operation=PDFOperation.EXTRACT_TEXT,
                error_message=f"Text extraction error: {str(e)}"
            )
    
    def extract_images(
        self,
        pdf_path: str,
        output_dir: str = "./pdf_images",
        image_format: ImageFormat = ImageFormat.PNG,
        pages: Optional[List[int]] = None
    ) -> PDFOperationResult:
        """
        Extract images from PDF.
        
        Args:
            pdf_path: Path to PDF file
            output_dir: Directory to save images
            image_format: Output image format
            pages: Specific pages to extract from (1-indexed)
            
        Returns:
            PDFOperationResult with extracted image paths
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            extracted_images = []
            
            with open(pdf_path, 'rb') as file:
                reader = pypdf.PdfReader(file)
                
                pages_to_process = pages if pages else range(1, len(reader.pages) + 1)
                
                for page_num in pages_to_process:
                    if page_num > len(reader.pages):
                        continue
                    
                    page = reader.pages[page_num - 1]
                    
                    if '/XObject' in page['/Resources']:
                        xobjects = page['/Resources']['/XObject'].get_object()
                        
                        for obj_name in xobjects:
                            if xobjects[obj_name]['/Subtype'] == '/Image':
                                image_obj = xobjects[obj_name]
                                
                                # Extract image data
                                if '/Filter' in image_obj:
                                    if image_obj['/Filter'] == '/DCTDecode':
                                        # JPEG image
                                        data = image_obj._data
                                        img_path = os.path.join(
                                            output_dir,
                                            f"page{page_num}_{obj_name[1:]}.jpg"
                                        )
                                        with open(img_path, 'wb') as img_file:
                                            img_file.write(data)
                                        extracted_images.append(img_path)
            
            return PDFOperationResult(
                success=True,
                operation=PDFOperation.EXTRACT_IMAGES,
                data=extracted_images,
                message=f"Extracted {len(extracted_images)} images",
                metadata={'num_images': len(extracted_images)}
            )
            
        except Exception as e:
            self.logger.error(f"Error extracting images: {str(e)}")
            return PDFOperationResult(
                success=False,
                operation=PDFOperation.EXTRACT_IMAGES,
                error_message=f"Image extraction error: {str(e)}"
            )
    
    def merge_pdfs(
        self,
        pdf_files: List[str],
        output_path: str,
        preserve_bookmarks: bool = True
    ) -> PDFOperationResult:
        """
        Merge multiple PDF files.
        
        Args:
            pdf_files: List of PDF file paths
            output_path: Output file path
            preserve_bookmarks: Preserve bookmarks from source PDFs
            
        Returns:
            PDFOperationResult
        """
        try:
            merger = pypdf.PdfMerger()
            
            for pdf_file in pdf_files:
                if os.path.exists(pdf_file):
                    merger.append(pdf_file, import_outline=preserve_bookmarks)
                else:
                    self.logger.warning(f"File not found: {pdf_file}")
            
            # Write merged PDF
            with open(output_path, 'wb') as output_file:
                merger.write(output_file)
            
            merger.close()
            
            return PDFOperationResult(
                success=True,
                operation=PDFOperation.MERGE,
                output_file=output_path,
                message=f"Merged {len(pdf_files)} PDF files",
                metadata={'num_files': len(pdf_files)}
            )
            
        except Exception as e:
            self.logger.error(f"Error merging PDFs: {str(e)}")
            return PDFOperationResult(
                success=False,
                operation=PDFOperation.MERGE,
                error_message=f"Merge error: {str(e)}"
            )
    
    def split_pdf(
        self,
        pdf_path: str,
        output_dir: str = "./split_pdfs",
        start_page: Optional[int] = None,
        end_page: Optional[int] = None,
        split_at_pages: Optional[List[int]] = None
    ) -> PDFOperationResult:
        """
        Split PDF into multiple files.
        
        Args:
            pdf_path: Path to PDF file
            output_dir: Directory for output files
            start_page: Start page for range split
            end_page: End page for range split
            split_at_pages: Split at specific pages
            
        Returns:
            PDFOperationResult with output file paths
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            output_files = []
            
            with open(pdf_path, 'rb') as file:
                reader = pypdf.PdfReader(file)
                base_name = os.path.splitext(os.path.basename(pdf_path))[0]
                
                if split_at_pages:
                    # Split at specific pages
                    split_points = [0] + sorted(split_at_pages) + [len(reader.pages)]
                    
                    for i in range(len(split_points) - 1):
                        writer = pypdf.PdfWriter()
                        
                        for page_num in range(split_points[i], split_points[i + 1]):
                            if page_num < len(reader.pages):
                                writer.add_page(reader.pages[page_num])
                        
                        output_file = os.path.join(
                            output_dir,
                            f"{base_name}_part{i+1}.pdf"
                        )
                        
                        with open(output_file, 'wb') as out_file:
                            writer.write(out_file)
                        
                        output_files.append(output_file)
                
                else:
                    # Extract page range
                    writer = pypdf.PdfWriter()
                    
                    start_idx = (start_page - 1) if start_page else 0
                    end_idx = end_page if end_page else len(reader.pages)
                    
                    for i in range(start_idx, min(end_idx, len(reader.pages))):
                        writer.add_page(reader.pages[i])
                    
                    output_file = os.path.join(
                        output_dir,
                        f"{base_name}_pages{start_idx+1}-{end_idx}.pdf"
                    )
                    
                    with open(output_file, 'wb') as out_file:
                        writer.write(out_file)
                    
                    output_files.append(output_file)
            
            return PDFOperationResult(
                success=True,
                operation=PDFOperation.SPLIT,
                data=output_files,
                message=f"Split into {len(output_files)} files",
                metadata={'num_files': len(output_files)}
            )
            
        except Exception as e:
            self.logger.error(f"Error splitting PDF: {str(e)}")
            return PDFOperationResult(
                success=False,
                operation=PDFOperation.SPLIT,
                error_message=f"Split error: {str(e)}"
            )
    
    def rotate_pages(
        self,
        pdf_path: str,
        output_path: str,
        rotation: int = 90,
        pages: Optional[List[int]] = None
    ) -> PDFOperationResult:
        """
        Rotate PDF pages.
        
        Args:
            pdf_path: Path to PDF file
            output_path: Output file path
            rotation: Rotation angle (90, 180, 270)
            pages: Specific pages to rotate (1-indexed)
            
        Returns:
            PDFOperationResult
        """
        try:
            with open(pdf_path, 'rb') as file:
                reader = pypdf.PdfReader(file)
                writer = pypdf.PdfWriter()
                
                for i, page in enumerate(reader.pages):
                    if pages is None or (i + 1) in pages:
                        page.rotate(rotation)
                    writer.add_page(page)
                
                with open(output_path, 'wb') as out_file:
                    writer.write(out_file)
            
            return PDFOperationResult(
                success=True,
                operation=PDFOperation.ROTATE,
                output_file=output_path,
                message=f"Rotated pages by {rotation} degrees"
            )
            
        except Exception as e:
            self.logger.error(f"Error rotating pages: {str(e)}")
            return PDFOperationResult(
                success=False,
                operation=PDFOperation.ROTATE,
                error_message=f"Rotation error: {str(e)}"
            )
    
    def add_watermark(
        self,
        pdf_path: str,
        output_path: str,
        watermark_text: str,
        opacity: float = 0.3,
        angle: int = 45,
        font_size: int = 50
    ) -> PDFOperationResult:
        """
        Add text watermark to PDF.
        
        Args:
            pdf_path: Path to PDF file
            output_path: Output file path
            watermark_text: Watermark text
            opacity: Watermark opacity (0-1)
            angle: Watermark angle
            font_size: Font size
            
        Returns:
            PDFOperationResult
        """
        try:
            # Create watermark PDF
            watermark_pdf = os.path.join(self.temp_dir, "watermark.pdf")
            c = canvas.Canvas(watermark_pdf, pagesize=letter)
            
            # Set watermark properties
            c.setFont("Helvetica", font_size)
            c.setFillAlpha(opacity)
            
            # Calculate position
            c.translate(300, 400)
            c.rotate(angle)
            
            # Draw watermark
            c.drawCentredString(0, 0, watermark_text)
            c.save()
            
            # Apply watermark
            with open(pdf_path, 'rb') as file:
                reader = pypdf.PdfReader(file)
                watermark_reader = pypdf.PdfReader(watermark_pdf)
                watermark_page = watermark_reader.pages[0]
                
                writer = pypdf.PdfWriter()
                
                for page in reader.pages:
                    page.merge_page(watermark_page)
                    writer.add_page(page)
                
                with open(output_path, 'wb') as out_file:
                    writer.write(out_file)
            
            # Clean up
            os.remove(watermark_pdf)
            
            return PDFOperationResult(
                success=True,
                operation=PDFOperation.WATERMARK,
                output_file=output_path,
                message=f"Added watermark: {watermark_text}"
            )
            
        except Exception as e:
            self.logger.error(f"Error adding watermark: {str(e)}")
            return PDFOperationResult(
                success=False,
                operation=PDFOperation.WATERMARK,
                error_message=f"Watermark error: {str(e)}"
            )
    
    def encrypt_pdf(
        self,
        pdf_path: str,
        output_path: str,
        user_password: str,
        owner_password: Optional[str] = None,
        use_128bit: bool = True
    ) -> PDFOperationResult:
        """
        Encrypt PDF with password.
        
        Args:
            pdf_path: Path to PDF file
            output_path: Output file path
            user_password: Password for opening PDF
            owner_password: Password for editing PDF
            use_128bit: Use 128-bit encryption
            
        Returns:
            PDFOperationResult
        """
        try:
            with open(pdf_path, 'rb') as file:
                reader = pypdf.PdfReader(file)
                writer = pypdf.PdfWriter()
                
                # Copy pages
                for page in reader.pages:
                    writer.add_page(page)
                
                # Set encryption
                writer.encrypt(
                    user_password=user_password,
                    owner_password=owner_password or user_password,
                    use_128bit=use_128bit
                )
                
                # Write encrypted PDF
                with open(output_path, 'wb') as out_file:
                    writer.write(out_file)
            
            return PDFOperationResult(
                success=True,
                operation=PDFOperation.ENCRYPT,
                output_file=output_path,
                message="PDF encrypted successfully"
            )
            
        except Exception as e:
            self.logger.error(f"Error encrypting PDF: {str(e)}")
            return PDFOperationResult(
                success=False,
                operation=PDFOperation.ENCRYPT,
                error_message=f"Encryption error: {str(e)}"
            )
    
    def decrypt_pdf(
        self,
        pdf_path: str,
        output_path: str,
        password: str
    ) -> PDFOperationResult:
        """
        Decrypt password-protected PDF.
        
        Args:
            pdf_path: Path to encrypted PDF
            output_path: Output file path
            password: PDF password
            
        Returns:
            PDFOperationResult
        """
        try:
            with open(pdf_path, 'rb') as file:
                reader = pypdf.PdfReader(file)
                
                # Decrypt
                if reader.is_encrypted:
                    if not reader.decrypt(password):
                        return PDFOperationResult(
                            success=False,
                            operation=PDFOperation.DECRYPT,
                            error_message="Invalid password"
                        )
                
                writer = pypdf.PdfWriter()
                
                # Copy pages
                for page in reader.pages:
                    writer.add_page(page)
                
                # Write decrypted PDF
                with open(output_path, 'wb') as out_file:
                    writer.write(out_file)
            
            return PDFOperationResult(
                success=True,
                operation=PDFOperation.DECRYPT,
                output_file=output_path,
                message="PDF decrypted successfully"
            )
            
        except Exception as e:
            self.logger.error(f"Error decrypting PDF: {str(e)}")
            return PDFOperationResult(
                success=False,
                operation=PDFOperation.DECRYPT,
                error_message=f"Decryption error: {str(e)}"
            )
    
    def ocr_pdf(
        self,
        pdf_path: str,
        output_path: Optional[str] = None,
        language: str = 'eng',
        dpi: int = 300
    ) -> PDFOperationResult:
        """
        Perform OCR on scanned PDF.
        
        Args:
            pdf_path: Path to PDF file
            output_path: Output text file path
            language: OCR language code
            dpi: DPI for image conversion
            
        Returns:
            PDFOperationResult with OCR text
        """
        if not OCR_AVAILABLE:
            return PDFOperationResult(
                success=False,
                operation=PDFOperation.OCR,
                error_message="OCR not available. Install pytesseract and tesseract-ocr."
            )
        
        try:
            ocr_text = []
            
            # Convert PDF pages to images
            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    # Convert page to image
                    img = page.to_image(resolution=dpi)
                    pil_image = img.original
                    
                    # Perform OCR
                    text = pytesseract.image_to_string(pil_image, lang=language)
                    ocr_text.append(f"=== Page {i+1} ===\n{text}")
            
            full_text = "\n\n".join(ocr_text)
            
            # Save to file if requested
            if output_path:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(full_text)
            
            return PDFOperationResult(
                success=True,
                operation=PDFOperation.OCR,
                data=full_text,
                output_file=output_path,
                message=f"OCR completed for {len(ocr_text)} pages"
            )
            
        except Exception as e:
            self.logger.error(f"Error during OCR: {str(e)}")
            return PDFOperationResult(
                success=False,
                operation=PDFOperation.OCR,
                error_message=f"OCR error: {str(e)}"
            )
    
    def compress_pdf(
        self,
        pdf_path: str,
        output_path: str,
        remove_duplication: bool = True,
        remove_images: bool = False,
        image_quality: int = 85
    ) -> PDFOperationResult:
        """
        Compress PDF file size.
        
        Args:
            pdf_path: Path to PDF file
            output_path: Output file path
            remove_duplication: Remove duplicate objects
            remove_images: Remove all images
            image_quality: JPEG quality (1-100)
            
        Returns:
            PDFOperationResult
        """
        try:
            with open(pdf_path, 'rb') as file:
                reader = pypdf.PdfReader(file)
                writer = pypdf.PdfWriter()
                
                for page in reader.pages:
                    # Compress page
                    page.compress_content_streams()
                    
                    if remove_images:
                        # Remove images from page
                        if '/XObject' in page['/Resources']:
                            xobjects = page['/Resources']['/XObject']
                            for obj_name in list(xobjects.keys()):
                                if xobjects[obj_name]['/Subtype'] == '/Image':
                                    del xobjects[obj_name]
                    
                    writer.add_page(page)
                
                # Remove duplication
                if remove_duplication:
                    writer.compress_identical_objects()
                
                # Write compressed PDF
                with open(output_path, 'wb') as out_file:
                    writer.write(out_file)
            
            # Calculate compression ratio
            original_size = os.path.getsize(pdf_path)
            compressed_size = os.path.getsize(output_path)
            compression_ratio = (1 - compressed_size / original_size) * 100
            
            return PDFOperationResult(
                success=True,
                operation=PDFOperation.COMPRESS,
                output_file=output_path,
                message=f"Compressed by {compression_ratio:.1f}%",
                metadata={
                    'original_size': original_size,
                    'compressed_size': compressed_size,
                    'compression_ratio': compression_ratio
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error compressing PDF: {str(e)}")
            return PDFOperationResult(
                success=False,
                operation=PDFOperation.COMPRESS,
                error_message=f"Compression error: {str(e)}"
            )
    
    def download_pdf(
        self,
        url: str,
        output_path: Optional[str] = None,
        chunk_size: int = 8192
    ) -> PDFOperationResult:
        """
        Download PDF from URL.
        
        Args:
            url: PDF URL
            output_path: Output file path
            chunk_size: Download chunk size
            
        Returns:
            PDFOperationResult
        """
        try:
            if not output_path:
                # Generate filename from URL
                filename = url.split('/')[-1]
                if not filename.endswith('.pdf'):
                    filename = 'downloaded.pdf'
                output_path = filename
            
            self.logger.info(f"Downloading PDF from: {url}")
            
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # Write PDF
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
            
            # Verify it's a valid PDF
            info = self.get_pdf_info(output_path)
            if not info:
                os.remove(output_path)
                return PDFOperationResult(
                    success=False,
                    operation=PDFOperation.EXTRACT_METADATA,
                    error_message="Downloaded file is not a valid PDF"
                )
            
            return PDFOperationResult(
                success=True,
                operation=PDFOperation.EXTRACT_METADATA,
                output_file=output_path,
                message=f"Downloaded PDF: {info.num_pages} pages, {info.file_size} bytes",
                data=info
            )
            
        except Exception as e:
            self.logger.error(f"Error downloading PDF: {str(e)}")
            return PDFOperationResult(
                success=False,
                operation=PDFOperation.EXTRACT_METADATA,
                error_message=f"Download error: {str(e)}"
            )


class PDFToolWrapper:
    """
    LLM Agent Tool wrapper for PDF operations.
    
    This class provides a simplified interface designed for use in LLM agent
    frameworks like LangChain, AutoGPT, etc.
    """
    
    def __init__(self):
        """Initialize the tool wrapper."""
        self.pdf_tool = PDFTool()
        self.name = "pdf_tool"
        self.description = (
            "Comprehensive PDF manipulation tool. Can read, merge, split, "
            "rotate, watermark, encrypt, decrypt, compress, and perform OCR on PDFs."
        )
    
    def run(self, operation: str, **kwargs) -> Dict[str, Any]:
        """
        Run PDF operation.
        
        Args:
            operation: Operation type
            **kwargs: Operation-specific arguments
            
        Returns:
            Dictionary with operation results
        """
        op_map = {
            'read': self.pdf_tool.extract_text,
            'extract_text': self.pdf_tool.extract_text,
            'extract_images': self.pdf_tool.extract_images,
            'merge': self.pdf_tool.merge_pdfs,
            'split': self.pdf_tool.split_pdf,
            'rotate': self.pdf_tool.rotate_pages,
            'watermark': self.pdf_tool.add_watermark,
            'encrypt': self.pdf_tool.encrypt_pdf,
            'decrypt': self.pdf_tool.decrypt_pdf,
            'compress': self.pdf_tool.compress_pdf,
            'ocr': self.pdf_tool.ocr_pdf,
            'download': self.pdf_tool.download_pdf,
            'info': lambda pdf_path: self.pdf_tool.get_pdf_info(pdf_path)
        }
        
        if operation not in op_map:
            return {
                'success': False,
                'error': f"Unknown operation: {operation}"
            }
        
        # Execute operation
        func = op_map[operation]
        
        if operation == 'info':
            info = func(kwargs.get('pdf_path'))
            if info:
                return {
                    'success': True,
                    'info': {
                        'filename': info.filename,
                        'num_pages': info.num_pages,
                        'file_size': info.file_size,
                        'title': info.title,
                        'author': info.author,
                        'encrypted': info.encrypted
                    }
                }
            else:
                return {'success': False, 'error': 'Failed to get PDF info'}
        else:
            result = func(**kwargs)
            return {
                'success': result.success,
                'operation': result.operation.value,
                'output_file': result.output_file,
                'message': result.message,
                'data': result.data,
                'error': result.error_message
            }


@register_tool(
    tags=["pdf", "document", "text", "merge", "split", "download"]
)
def download_pdf_from_url(
    url: str,
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """Downloads a PDF file from any URL.
    
    Args:
        url: URL of the PDF file
        output_path: Where to save the PDF (optional)
        
    Returns:
        Dictionary with download status and file info
    """
    pdf_tool = PDFTool()
    result = pdf_tool.download_pdf(url, output_path)
    
    if result.success and result.data:
        info = result.data
        return {
            'success': True,
            'output_path': result.output_file,
            'num_pages': info.num_pages,
            'file_size': info.file_size,
            'title': info.title,
            'message': result.message
        }
    else:
        return {
            'success': False,
            'error': result.error_message
        }


@register_tool(
    tags=["pdf", "document", "text", "merge", "split"]
)
def pdf_tool(
    operation: str,
    pdf_path: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """Performs various operations on PDF files.
    
    Args:
        operation: Operation to perform (read, merge, split, rotate, etc.)
        pdf_path: Path to PDF file (for single-file operations)
        **kwargs: Additional operation-specific arguments
        
    Returns:
        Dictionary with operation results
    """
    tool = PDFToolWrapper()
    if pdf_path:
        kwargs['pdf_path'] = pdf_path
    return tool.run(operation, **kwargs)


# Example usage and testing
if __name__ == "__main__":
    print("=== PDF Tool Test ===\n")
    
    # List PDF files in current directory
    pdf_files = glob.glob("*.pdf")
    
    if not pdf_files:
        print("No PDF files found in current directory.")
        print("\nYou can:")
        print("1. Download a PDF from URL")
        print("2. Exit")
        
        choice = input("\nSelect option: ").strip()
        
        if choice == "1":
            url = input("Enter PDF URL: ").strip()
            tool = PDFTool()
            result = tool.download_pdf(url, "downloaded.pdf")
            if result.success:
                print(f"Downloaded: {result.output_file}")
                pdf_files = ["downloaded.pdf"]
            else:
                print(f"Error: {result.error_message}")
                sys.exit(1)
        else:
            sys.exit(0)
    
    print("\nPDF files found:")
    for i, file in enumerate(pdf_files, 1):
        print(f"{i}. {file}")
    
    # Select file
    file_choice = input("\nSelect PDF file number: ").strip()
    try:
        selected_pdf = pdf_files[int(file_choice) - 1]
    except:
        print("Invalid selection")
        sys.exit(1)
    
    # Operation menu
    print(f"\nSelected: {selected_pdf}")
    print("\nOperations:")
    print("1. Extract text")
    print("2. Get PDF info")
    print("3. Split PDF")
    print("4. Merge with another PDF")
    print("5. Add watermark")
    print("6. Compress PDF")
    print("7. Extract images")
    print("8. Rotate pages")
    print("9. Encrypt PDF")
    print("10. OCR (for scanned PDFs)")
    
    op_choice = input("\nSelect operation: ").strip()
    
    tool = PDFTool()
    
    if op_choice == "1":
        # Extract text
        print("\nExtracting text...")
        result = tool.extract_text(selected_pdf)
        if result.success:
            print(f"\n{result.data[:1000]}...")
            if len(result.data) > 1000:
                print(f"\n... and {len(result.data) - 1000} more characters")
            
            save = input("\nSave to file? (y/n): ").strip().lower()
            if save == 'y':
                output_file = selected_pdf.replace('.pdf', '_text.txt')
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(result.data)
                print(f"Saved to: {output_file}")
        else:
            print(f"Error: {result.error_message}")
    
    elif op_choice == "2":
        # Get info
        info = tool.get_pdf_info(selected_pdf)
        if info:
            print(f"\nPDF Information:")
            print(f"Filename: {info.filename}")
            print(f"Pages: {info.num_pages}")
            print(f"Size: {info.file_size:,} bytes")
            print(f"Title: {info.title or 'N/A'}")
            print(f"Author: {info.author or 'N/A'}")
            print(f"Encrypted: {info.encrypted}")
    
    elif op_choice == "3":
        # Split PDF
        print("\nSplit options:")
        print("1. Extract page range")
        print("2. Split at specific pages")
        
        split_choice = input("Select option: ").strip()
        
        if split_choice == "1":
            start = input("Start page: ").strip()
            end = input("End page: ").strip()
            
            result = tool.split_pdf(
                selected_pdf,
                start_page=int(start) if start else None,
                end_page=int(end) if end else None
            )
        else:
            pages = input("Split at pages (comma-separated): ").strip()
            split_pages = [int(p.strip()) for p in pages.split(',')]
            
            result = tool.split_pdf(
                selected_pdf,
                split_at_pages=split_pages
            )
        
        if result.success:
            print(f"\nCreated files:")
            for file in result.data:
                print(f"  - {file}")
        else:
            print(f"Error: {result.error_message}")
    
    else:
        print("Operation not implemented in interactive mode")