"""
build_paper.py - paper.md -> VaniStream_Paper.docx (editable, native Word equations) -> VaniStream_Paper.pdf

  python research_paper/scripts/build_paper.py

Requires: pypandoc_binary (bundles pandoc), python-docx; PDF export uses Microsoft Word via COM (Windows).
"""
import os
import subprocess
import sys

import pypandoc
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor, Cm

PAPER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(PAPER, "paper.md")
REF = os.path.join(PAPER, "scripts", "reference.docx")
DOCX = os.path.join(PAPER, "VaniStream_Paper.docx")
PDF = os.path.join(PAPER, "VaniStream_Paper.pdf")
BODY, HEAD = "Times New Roman", "Arial"


def make_reference():
    """Pandoc's default reference document, restyled (fonts, sizes, spacing, captions)."""
    subprocess.run([pypandoc.get_pandoc_path(), "-o", REF, "--print-default-data-file", "reference.docx"], check=True,
                   stdout=open(REF, "wb"))
    d = Document(REF)
    st = d.styles
    def font(name, face, size, bold=None, italic=None, color=None, before=None, after=None, align=None, line=None):
        s = st[name]
        s.font.name = face
        s.element.rPr.rFonts.set(qn("w:eastAsia"), face) if s.element.rPr is not None and s.element.rPr.rFonts is not None else None
        s.font.size = Pt(size)
        if bold is not None:
            s.font.bold = bold
        if italic is not None:
            s.font.italic = italic
        s.font.color.rgb = RGBColor.from_string(color or "000000")
        pf = s.paragraph_format
        if before is not None:
            pf.space_before = Pt(before)
        if after is not None:
            pf.space_after = Pt(after)
        if align is not None:
            pf.alignment = align
        if line is not None:
            pf.line_spacing = line
    for n in ("Normal", "Body Text", "First Paragraph", "Compact"):
        if n in [x.name for x in st]:
            font(n, BODY, 11, after=6, line=1.15)
    font("Title", HEAD, 17, bold=True, after=12, align=WD_ALIGN_PARAGRAPH.LEFT)
    font("Heading 1", HEAD, 13.5, bold=True, before=16, after=6)
    font("Heading 2", HEAD, 11.5, bold=True, before=10, after=4)
    font("Heading 3", HEAD, 11, bold=True, italic=True, before=8, after=3)
    for n in ("Image Caption", "Table Caption", "Caption"):
        if n in [x.name for x in st]:
            font(n, BODY, 9.5, italic=False, after=10, align=WD_ALIGN_PARAGRAPH.LEFT, line=1.0)
    d.save(REF)


def set_cell_border(table):
    tbl = table._tbl
    pr = tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "bottom", "insideH"):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), "8" if edge != "insideH" else "4")
        el.set(qn("w:color"), "808080")
        borders.append(el)
    for old in pr.findall(qn("w:tblBorders")):
        pr.remove(old)
    pr.append(borders)


def postprocess():
    d = Document(DOCX)
    sec = d.sections[0]
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
    for side in ("left_margin", "right_margin"):
        setattr(sec, side, Cm(2.3))
    sec.top_margin = sec.bottom_margin = Cm(2.3)
    # continuous line numbers in the margin, as in the PeerJ submission format
    ln = OxmlElement("w:lnNumType")
    ln.set(qn("w:countBy"), "1")
    ln.set(qn("w:restart"), "continuous")
    ln.set(qn("w:distance"), "283")
    sec._sectPr.append(ln)
    # page numbers in the footer
    p = sec.footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for kind, text in (("begin", None), (None, "PAGE"), ("end", None)):
        r = p.add_run()
        if kind:
            fc = OxmlElement("w:fldChar")
            fc.set(qn("w:fldCharType"), kind)
            r._r.append(fc)
        else:
            it = OxmlElement("w:instrText")
            it.set(qn("xml:space"), "preserve")
            it.text = text
            r._r.append(it)
    for t in d.tables:
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        set_cell_border(t)
        for i, row in enumerate(t.rows):
            for cell in row.cells:
                for para in cell.paragraphs:
                    para.paragraph_format.space_after = Pt(1)
                    para.paragraph_format.line_spacing = 1.0
                    for run in para.runs:
                        run.font.size = Pt(9)
                        run.font.name = BODY
                        if i == 0:
                            run.font.bold = True
    for para in d.paragraphs:
        if para._p.findall(".//" + qn("w:drawing")):
            para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            para.paragraph_format.keep_with_next = True
        if para.style.name in ("Table Caption",):
            para.paragraph_format.keep_with_next = True
    d.core_properties.title = "VaniStream: Source-Grounded Conversational Retrieval over Transcribed All India Radio Bulletins and NewsOnAIR Articles"
    d.core_properties.author = "Varad More; Manav Jadhav; Soham Jadhav; Anant Nimkar"
    d.save(DOCX)


def to_pdf():
    ps = (f"$w = New-Object -ComObject Word.Application; $w.Visible = $false; "
          f"$d = $w.Documents.Open('{DOCX}', $false, $true); "
          f"$d.ExportAsFixedFormat('{PDF}', 17); $d.Close($false); $w.Quit()")
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True)


def main():
    make_reference()
    pypandoc.convert_file(SRC, "docx", outputfile=DOCX,
                          extra_args=["--number-sections", f"--reference-doc={REF}", f"--resource-path={PAPER}"])
    postprocess()
    print("wrote", DOCX)
    if "--no-pdf" not in sys.argv:
        to_pdf()
        print("wrote", PDF)


if __name__ == "__main__":
    main()
