"""SmartArt text extraction from PPTX files.

python-pptx does not expose SmartArt content — those shapes appear as
empty GraphicFrame elements. The actual text lives in the diagram XML
parts inside the PPTX ZIP archive:

    ppt/diagrams/dataN.xml  ->  <dgm:pt> elements with <a:t> text nodes

This module parses the diagram XML directly via lxml and maps diagram
parts back to slides via the relationship files.
"""

from __future__ import annotations

import logging
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

logger = logging.getLogger(__name__)

# OOXML namespaces used in diagram/relationship XML
_NS = {
    "dgm": "http://schemas.openxmlformats.org/drawingml/2006/diagram",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "rel": "http://schemas.openxmlformats.org/package/2006/relationships",
}


def extract_smartart_text(pptx_path: Path) -> dict[int, list[str]]:
    """Extract text from SmartArt diagrams in a PPTX file.

    Returns a dict mapping slide index (0-based) to a list of text strings
    found in SmartArt shapes on that slide.

    Args:
        pptx_path: Path to the .pptx file.

    Returns:
        {slide_index: [text1, text2, ...]}
    """
    result: dict[int, list[str]] = {}

    try:
        with zipfile.ZipFile(pptx_path, "r") as zf:
            names = set(zf.namelist())

            # 1. Find all diagram data files
            diagram_files = sorted(
                n for n in names if n.startswith("ppt/diagrams/data") and n.endswith(".xml")
            )
            if not diagram_files:
                return result

            # 2. Build a map: diagram data filename -> slide index
            diagram_to_slide = _map_diagrams_to_slides(zf, names)

            # 3. Extract text from each diagram data file
            for diag_file in diagram_files:
                slide_idx = diagram_to_slide.get(diag_file)
                if slide_idx is None:
                    # Try to match by just the base name
                    base = diag_file.rsplit("/", 1)[-1]
                    for k, v in diagram_to_slide.items():
                        if k.endswith(base):
                            slide_idx = v
                            break

                if slide_idx is None:
                    logger.debug("Could not map %s to a slide, skipping", diag_file)
                    continue

                texts = _extract_diagram_texts(zf, diag_file)
                if texts:
                    result.setdefault(slide_idx, []).extend(texts)

    except zipfile.BadZipFile:
        logger.warning("Cannot read %s as ZIP for SmartArt extraction", pptx_path)
    except Exception:
        logger.exception("SmartArt extraction failed for %s", pptx_path)

    return result


def _map_diagrams_to_slides(zf: zipfile.ZipFile, names: set[str]) -> dict[str, int]:
    """Build a mapping from diagram data filenames to slide indices.

    Walks ppt/slides/_rels/slideN.xml.rels to find relationships
    pointing to ../diagrams/dataN.xml.
    """
    mapping: dict[str, int] = {}

    slide_rels = sorted(
        n for n in names if n.startswith("ppt/slides/_rels/slide") and n.endswith(".xml.rels")
    )

    for rels_file in slide_rels:
        # Extract slide number from filename: slide1.xml.rels -> 0
        base = rels_file.rsplit("/", 1)[-1]  # slide1.xml.rels
        slide_name = base.replace(".xml.rels", "")  # slide1
        try:
            slide_num = int(slide_name.replace("slide", "")) - 1
        except ValueError:
            continue

        try:
            rels_xml = zf.read(rels_file)
            root = ET.fromstring(rels_xml)

            for rel in root.findall("rel:Relationship", _NS):
                target = rel.get("Target", "")
                # Targets look like: ../diagrams/data1.xml
                if "/diagrams/data" in target:
                    # Normalize to full path
                    if target.startswith(".."):
                        full_path = "ppt/" + target.removeprefix("../").removeprefix("../")
                    else:
                        full_path = target
                    # Normalize path separators
                    full_path = full_path.replace("\\", "/")
                    mapping[full_path] = slide_num

        except Exception:
            logger.debug("Failed to parse rels file %s", rels_file)

    return mapping


def _extract_diagram_texts(zf: zipfile.ZipFile, diag_file: str) -> list[str]:
    """Extract all text strings from a diagram data XML file.

    SmartArt text lives in <dgm:pt> (point) elements, each containing
    <dgm:t> -> <a:p> -> <a:r> -> <a:t> text runs.
    """
    try:
        xml_data = zf.read(diag_file)
        root = ET.fromstring(xml_data)
    except Exception:
        logger.debug("Failed to parse diagram XML %s", diag_file)
        return []

    texts: list[str] = []

    # Find all text runs in the diagram. SmartArt text is nested as:
    # <dgm:ptLst> / <dgm:pt> / <dgm:t> / <a:bodyPr>, <a:p> / <a:r> / <a:t>
    # But the namespace structure varies. Search broadly for <a:t> elements.
    for at_elem in root.iter(f"{{{_NS['a']}}}t"):
        text = (at_elem.text or "").strip()
        if text:
            texts.append(text)

    return texts
