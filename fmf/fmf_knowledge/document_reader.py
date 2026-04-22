"""
Document Reader для FMF - чтение текстовых файлов
Адаптировано из EVA-Ai
"""
import os
import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger("fmf.document_reader")


SUPPORTED_EXTENSIONS = {'.txt', '.md', '.log', '.json', '.xml', '.csv', '.yaml', '.yml', '.py', '.js', '.c', '.h'}


@dataclass
class DocumentContent:
    """Содержимое документа."""
    filename: str
    filepath: str
    content: str
    lines: List[str]
    metadata: Dict[str, Any]


class FMFDocumentReader:
    """
    Читает текстовые файлы.
    Поддерживает: .txt, .md, .log, .json, .xml, .csv, .yaml, .yml
    """
    
    def __init__(self, max_chars: int = 100000):
        self.max_chars = max_chars
    
    def read(self, filepath: str) -> Optional[DocumentContent]:
        """Читает файл."""
        if not os.path.exists(filepath):
            logger.error(f"Файл не найден: {filepath}")
            return None
        
        ext = os.path.splitext(filepath)[1].lower()
        
        if ext not in SUPPORTED_EXTENSIONS:
            logger.warning(f"Неподдерживаемый формат: {ext}")
            return None
        
        try:
            return self._read_file(filepath, ext)
        except Exception as e:
            logger.error(f"Ошибка чтения: {e}")
            return None
    
    def _read_file(self, filepath: str, ext: str) -> DocumentContent:
        """Внутренний метод чтения."""
        filename = os.path.basename(filepath)
        
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        
        # Ограничиваем размер
        if len(content) > self.max_chars:
            content = content[:self.max_chars]
        
        lines = content.split('\n')
        
        return DocumentContent(
            filename=filename,
            filepath=filepath,
            content=content,
            lines=lines,
            metadata={
                'size': len(content),
                'lines': len(lines),
                'extension': ext
            }
        )
    
    def read_multiple(self, filepaths: List[str]) -> List[DocumentContent]:
        """Читает несколько файлов."""
        results = []
        for fp in filepaths:
            doc = self.read(fp)
            if doc:
                results.append(doc)
        return results