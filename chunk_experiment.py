"""
Extract text from a PDF and split it into chunks
using RecursiveCharacterTextSplitter.
"""

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def extract_chapter_text(pdf_path: str) -> str:
    """Extract text from all pages of the PDF."""

    reader = PdfReader(pdf_path)

    text = ""

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

        print(f"Processed page {page_number}")

    return text


def split_by_section_with_limits(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> list[str]:
    """Split extracted text into chunks."""

    text_splitter = RecursiveCharacterTextSplitter(
        separators=[
            r"\n(?=\d+\.\d+\s)", ". "
            "\n\n",
            "\n",
            " ",
            "",
        ],
        is_separator_regex=True,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks = text_splitter.split_text(text)

    return chunks


if __name__ == "__main__":

    # --------------------------------------------------
    # 1. PDF path
    # --------------------------------------------------
    pdf_path = "knowledge/chapter_2.pdf"

    # --------------------------------------------------
    # 2. Extract text from PDF
    # --------------------------------------------------
    chapter_text = extract_chapter_text(pdf_path)

    print("\n" + "=" * 80)
    print(f"Total characters extracted = {len(chapter_text)}")
    print("=" * 80)

    # --------------------------------------------------
    # 3. Check extracted text
    # --------------------------------------------------
    if not chapter_text.strip():
        print("ERROR: No text was extracted from the PDF.")
        exit()

    # --------------------------------------------------
    # 4. Create chunks
    # --------------------------------------------------
    chunks = split_by_section_with_limits(
        chapter_text,
        chunk_size=1000,
        chunk_overlap=150,
    )

    # --------------------------------------------------
    # 5. Chunk information
    # --------------------------------------------------
    print(f"\nTotal number of chunks = {len(chunks)}")

    # --------------------------------------------------
    # 6. Check first chunk
    # --------------------------------------------------
    if len(chunks) > 0:
        print(f"First chunk length = {len(chunks[0])}")

        print("\n" + "=" * 80)
        print("FIRST CHUNK")
        print("=" * 80)
        print(chunks[0])

    # --------------------------------------------------
    # 7. Check second chunk
    # --------------------------------------------------
    if len(chunks) > 1:
        print("\n" + "=" * 80)
        print("SECOND CHUNK")
        print("=" * 80)
        print(chunks[1])

    # --------------------------------------------------
    # 8. Print size of every chunk
    # --------------------------------------------------
    print("\n" + "=" * 80)
    print("CHUNK SIZES")
    print("=" * 80)

    for index, chunk in enumerate(chunks, start=1):
        print(f"Chunk {index}: {len(chunk)} characters")