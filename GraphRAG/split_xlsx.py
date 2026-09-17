#!/usr/bin/env python3

from pathlib import Path
import argparse
import pandas as pd

#-------------------------------------------------------------------------------
# Used to split *Metadata_Fields_Update.xlsx; could be used for other tasks.
#-------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Split an Excel workbook into one CSV file per worksheet, "
            "skipping the 'Prompt context' worksheet."
        )
    )

    parser.add_argument(
        "xlsx_file",
        help="Path to the source .xlsx file"
    )

    parser.add_argument(
        "csv_output_dir",
        help="Directory where CSV files will be written"
    )

    args = parser.parse_args()

    xlsx_file = Path(args.xlsx_file)
    output_dir = Path(args.csv_output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    xls = pd.ExcelFile(xlsx_file, engine="openpyxl")

    for sheet_name in xls.sheet_names:
        if sheet_name == "Prompt context":
            print(f"Skipping worksheet: {sheet_name}")
            continue

        print(f"Exporting worksheet: {sheet_name}")

        df = pd.read_excel(
            xlsx_file,
            sheet_name=sheet_name,
            engine="openpyxl",
            dtype=object
        )

        csv_file = output_dir / f"{sheet_name}.csv"

        df.to_csv(
            csv_file,
            index=False,
            encoding="utf-8",
            lineterminator="\n"
        )

    print("Done.")


if __name__ == "__main__":
    main()
