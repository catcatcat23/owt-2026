"""Package imagegen architecture PNGs as PDFs; does not modify source images."""
from pathlib import Path

import fitz


def main():
    figures = Path(__file__).resolve().parents[1] / "docs" / "figures"
    names = ("e_cross", "sam_soft_prior_aux", "sam_downsample_gt")
    combined = fitz.open()
    for name in names:
        image = fitz.open(figures / ("architecture_" + name + ".png"))
        pdf = fitz.open("pdf", image.convert_to_pdf())
        pdf.save(figures / ("architecture_" + name + ".pdf"))
        combined.insert_pdf(pdf)
        pdf.close()
        image.close()
    combined.set_metadata({"title": "OrganSlot: Three Retained Architectures"})
    combined.save(figures / "organslot_three_architectures.pdf")
    combined.close()


if __name__ == "__main__":
    main()
