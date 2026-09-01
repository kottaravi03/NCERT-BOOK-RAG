from extract import extract_pdf


def main():

    pdf_path = "knowledge/chapter_2.pdf"
    output_dir = "knowledge/extracted"

    extract_pdf(
        pdf_path,
        output_dir
    )


if __name__ == "__main__":
    main()