"""Generate the illustrative policy PDFs in policies/.

These are SAMPLE documents written for this project. They follow the general
shape of public RWBCIS term sheets and PMFBY guidelines but are not issued by
any insurer or government body and carry made-up numbers.

    uv run python scripts/make_sample_policies.py [out_dir]
"""

import sys
from datetime import UTC, datetime
from pathlib import Path

from fpdf import FPDF

FIXED_DATE = datetime(2023, 5, 1, tzinfo=UTC)

DISCLAIMER = (
    "SAMPLE DOCUMENT - ILLUSTRATIVE ONLY. Not issued by any insurer, bank or government "
    "body. Figures are invented for software testing and must not be used for real claims."
)

# policy_id -> (title, pages); each page is a list of (heading, paragraphs).
SAMPLES: dict[str, tuple[str, list[list[tuple[str, list[str]]]]]] = {
    "sample_rwbcis_paddy_thanjavur": (
        "Sample Weather Index Term Sheet: Paddy, Kharif 2023, Thanjavur",
        [
            [
                (
                    "Scope",
                    [
                        "Scheme: weather index based crop insurance, modelled on the "
                        "Restructured Weather Based Crop Insurance Scheme (RWBCIS).",
                        "Crop covered: Paddy (rice). Season: Kharif 2023.",
                        "Area: Thanjavur district, Tamil Nadu. Reference weather is the "
                        "gridded rainfall and temperature at the insured location.",
                        "Risk period: 1 June 2023 to 30 September 2023.",
                        "Sum insured: Rs. 60,000 per hectare. Farmer premium: 5 percent of "
                        "the sum insured. Enrolment closes on 15 July 2023.",
                    ],
                ),
                (
                    "1. Deficit rainfall (absolute)",
                    [
                        "1.1 Deficit rainfall cover applies when cumulative rainfall during "
                        "the risk period is less than 600 mm.",
                        "1.2 The payout rises linearly from the trigger of 600 mm to the exit "
                        "of 250 mm, at which the maximum payout of 40 percent of the sum "
                        "insured is reached.",
                    ],
                ),
            ],
            [
                (
                    "2. Deficit rainfall (relative to normal)",
                    [
                        "2.1 Additional deficit cover applies when the seasonal rainfall "
                        "deficit relative to the long-period normal is 40 percent or more.",
                        "2.2 The payout rises linearly to a maximum of 20 percent of the sum "
                        "insured at a deficit of 80 percent.",
                    ],
                ),
                (
                    "3. High temperature",
                    [
                        "3.1 High temperature cover applies when cumulative growing degree "
                        "days over the risk period exceed the normal by more than 15 percent.",
                        "3.2 The payout rises linearly to a maximum of 15 percent of the sum "
                        "insured at an excess of 30 percent.",
                    ],
                ),
                (
                    "4. General conditions",
                    [
                        "4.1 Claims are assessed from weather data alone; no field "
                        "inspection is needed. Total payout across all covers is capped at "
                        "100 percent of the sum insured.",
                        "4.2 Growing degree days are computed as the mean of daily maximum "
                        "and minimum temperature minus a base of 10 degrees Celsius, floored "
                        "at zero.",
                    ],
                ),
            ],
        ],
    ),
    "sample_pmfby_mid_season_adversity": (
        "Sample Yield Scheme Clause: Mid-Season Adversity, Kharif 2023",
        [
            [
                (
                    "Scope",
                    [
                        "Scheme: area yield based crop insurance, modelled on the "
                        "Pradhan Mantri Fasal Bima Yojana (PMFBY) operational guidelines.",
                        "Crops covered: Paddy, Maize and Cotton. Season: Kharif 2023, "
                        "1 June 2023 to 30 September 2023.",
                        "Sum insured: Rs. 55,000 per hectare for Paddy, Rs. 45,000 per "
                        "hectare for Maize and Rs. 70,000 per hectare for Cotton.",
                    ],
                ),
                (
                    "1. Prevented sowing",
                    [
                        "1.1 Where at least 75 percent of the sown area could not be sown "
                        "owing to deficit rainfall, an indemnity of up to 25 percent of the "
                        "sum insured is payable and cover ends.",
                    ],
                ),
                (
                    "2. Mid-season adversity",
                    [
                        "2.1 Mid-season adversity may be declared when the seasonal rainfall "
                        "deficit is 50 percent or more of the normal.",
                        "2.2 On declaration, an on-account payment of up to 25 percent of the "
                        "sum insured is made, adjusted against the final claim.",
                    ],
                ),
                (
                    "3. Final claim",
                    [
                        "3.1 The final claim is computed from crop cutting experiments as "
                        "(threshold yield - actual yield) / threshold yield x sum insured.",
                    ],
                ),
            ],
        ],
    ),
}


def build(out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for policy_id, (title, pages) in SAMPLES.items():
        pdf = FPDF()
        pdf.set_creation_date(FIXED_DATE)  # deterministic bytes for a stable index fingerprint
        pdf.set_title(title)
        for i, sections in enumerate(pages):
            pdf.add_page()
            if i == 0:
                pdf.set_font("Helvetica", "B", 14)
                pdf.multi_cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
                pdf.set_font("Helvetica", "I", 9)
                pdf.multi_cell(0, 5, DISCLAIMER, new_x="LMARGIN", new_y="NEXT")
                pdf.ln(4)
            for heading, paragraphs in sections:
                pdf.set_font("Helvetica", "B", 11)
                pdf.multi_cell(0, 7, heading, new_x="LMARGIN", new_y="NEXT")
                pdf.set_font("Helvetica", "", 10)
                for p in paragraphs:
                    pdf.multi_cell(0, 5, p, new_x="LMARGIN", new_y="NEXT")
                    pdf.ln(1)
                pdf.ln(2)
        path = out_dir / f"{policy_id}.pdf"
        pdf.output(str(path))
        paths.append(path)
    return paths


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parents[1] / "policies"
    for p in build(out):
        print(p)
