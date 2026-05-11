import PyPDF2
import docx
import tempfile
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple
from werkzeug.datastructures import FileStorage


def read_pdf(file_path: str) -> str:
    """Extract text from PDF file"""
    text = ""
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        raise ValueError(f"Error reading PDF: {str(e)}")
    return text


def read_docx(file_path: str) -> str:
    """Extract text from DOCX file"""
    try:
        doc = docx.Document(file_path)
        return "\n".join([p.text for p in doc.paragraphs])
    except Exception as e:
        raise ValueError(f"Error reading DOCX: {str(e)}")


def process_file(file: FileStorage) -> Tuple[str, str]:
    """
    Process uploaded file and extract text.
    Returns tuple of (filename, extracted_text)
    File is saved temporarily and deleted after processing.
    """
    # Create temp file
    temp_dir = tempfile.gettempdir()
    temp_path = None
    
    try:
        # Save file temporarily
        filename = file.filename
        if not filename:
            raise ValueError("Filename is required")
        
        # Determine file extension
        ext = filename.lower().split('.')[-1]
        
        # Save to temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{ext}', dir=temp_dir) as temp_file:
            file.save(temp_file.name)
            temp_path = temp_file.name
        
        # Extract text based on file type
        if ext == 'pdf':
            text = read_pdf(temp_path)
        elif ext in ['docx', 'doc']:
            text = read_docx(temp_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
        
        return filename, text
    
    finally:
        # Clean up temp file
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass  # Ignore cleanup errors


def process_multiple_files(files: List[FileStorage]) -> List[Tuple[str, str]]:
    """
    Process multiple uploaded files (sequential).
    Returns list of tuples (filename, extracted_text)
    """
    results = []
    for file in files:
        try:
            filename, text = process_file(file)
            results.append((filename, text))
        except Exception as e:
            # Continue processing other files even if one fails
            results.append((file.filename or "unknown", f"Error processing file: {str(e)}"))
    return results


def process_multiple_files_parallel(
    files: List[FileStorage], max_workers: int = 8
) -> List[Tuple[str, str]]:
    """
    Process multiple uploaded files in parallel.
    Returns list of tuples (filename, extracted_text) in the same order as input.
    """
    def process_one(idx_and_file: Tuple[int, FileStorage]) -> Tuple[int, Tuple[str, str]]:
        idx, file = idx_and_file
        try:
            filename, text = process_file(file)
            return (idx, (filename, text))
        except Exception as e:
            return (idx, (file.filename or "unknown", f"Error processing file: {str(e)}"))

    result_by_idx = {}
    with ThreadPoolExecutor(max_workers=min(max_workers, len(files))) as executor:
        futures = {
            executor.submit(process_one, (i, f)): i
            for i, f in enumerate(files)
        }
        for future in as_completed(futures):
            idx, pair = future.result()
            result_by_idx[idx] = pair

    return [result_by_idx[i] for i in range(len(files))]
