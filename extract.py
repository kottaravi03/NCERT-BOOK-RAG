import os
import fitz  # PyMuPDF


def extract_pdf(pdf_path, output_dir="./extracted"):
    """
    Extract only:
        1. Text from PDF pages
        2. Real embedded images

    Does NOT render complete PDF pages.

    Images are converted to RGB PNG to avoid:
        - black images
        - CMYK problems
        - JPX/JPEG2000 problems
        - transparency problems
        - image-mask problems
    """

    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}"
        )
    text_dir = os.path.join(output_dir, "text")
    image_dir = os.path.join(output_dir, "images")

    os.makedirs(text_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)
    doc = fitz.open(pdf_path)

    print("=" * 70)
    print(f"PDF   : {os.path.basename(pdf_path)}")
    print(f"Pages : {len(doc)}")
    print("=" * 70)

    extracted_xrefs = set()

    total_images = 0

    for page_number, page in enumerate(doc, start=1):

        print(
            f"\nProcessing page "
            f"{page_number}/{len(doc)}"
        )

        text = page.get_text("text").strip()

        text_file = os.path.join(
            text_dir,
            f"page_{page_number:03}.txt"
        )

        with open(
            text_file,
            "w",
            encoding="utf-8"
        ) as f:
            f.write(text)

        print("  ✓ Text extracted")

        images = page.get_images(full=True)

        if not images:
            print("  - No embedded images")
            continue

        page_image_count = 0

        for image_number, image_info in enumerate(
            images,
            start=1
        ):

            xref = image_info[0]

            if xref in extracted_xrefs:
                continue

            extracted_xrefs.add(xref)

            try:

                image_data = doc.extract_image(xref)

                width = image_data["width"]
                height = image_data["height"]

                print(
                    f"    Image {image_number}: "
                    f"{width}x{height}"
                )

                if width < 100 or height < 100:
                    print("      - Skipped: too small")
                    continue

                smask = image_info[1]

                if smask != 0:
                    print(
                        "      - Has transparency mask"
                    )

                pix = fitz.Pixmap(doc, xref)

                if pix.n - pix.alpha > 3:

                    pix = fitz.Pixmap(
                        fitz.csRGB,
                        pix
                    )

                elif pix.colorspace is None:

                    # Grayscale / mask
                    pix = fitz.Pixmap(
                        fitz.csRGB,
                        pix
                    )

                if pix.alpha:

                    pix = fitz.Pixmap(
                        fitz.csRGB,
                        pix
                    )

                image_file = os.path.join(
                    image_dir,
                    f"page_{page_number:03}"
                    f"_image_{page_image_count + 1:03}.png"
                )

                pix.save(image_file)

                pix = None

                page_image_count += 1
                total_images += 1

                print(
                    f"      ✓ Saved: "
                    f"{os.path.basename(image_file)}"
                )

            except Exception as e:

                print(
                    f"      ✗ Failed image "
                    f"{image_number}: {e}"
                )

        if page_image_count == 0:

            print("  - No usable images")

        else:

            print(
                f"  ✓ {page_image_count} "
                f"image(s) extracted"
            )

    doc.close()

    print("\n" + "=" * 70)
    print("EXTRACTION COMPLETED")
    print("=" * 70)

    print(f"Text   : {text_dir}")
    print(f"Images : {image_dir}")
    print(f"Total images : {total_images}")

    print("=" * 70)

if __name__ == "__main__":

    extract_pdf(
        pdf_path="knowledge/chapter_2.pdf",
        output_dir="knowledge/extracted"
    ) 