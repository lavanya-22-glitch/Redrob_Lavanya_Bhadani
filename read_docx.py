import zipfile
import xml.etree.ElementTree as ET
import sys
from pathlib import Path

def extract_text_from_docx(docx_path):
    namespaces = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
    with zipfile.ZipFile(docx_path) as docx:
        xml_content = docx.read('word/document.xml')
    tree = ET.fromstring(xml_content)
    paragraphs = []
    for paragraph in tree.iterfind('.//w:p', namespaces):
        texts = [node.text for node in paragraph.iterfind('.//w:t', namespaces) if node.text]
        if texts:
            paragraphs.append(''.join(texts))
    return '\n'.join(paragraphs)

info_dir = Path(r"c:\Lavi\Learn\Hackathons\Redorb\info")
for docx in info_dir.glob("*.docx"):
    text = extract_text_from_docx(docx)
    txt_path = docx.with_suffix(".txt")
    txt_path.write_text(text, encoding="utf-8")
    print(f"Extracted {docx.name} to {txt_path.name}")
