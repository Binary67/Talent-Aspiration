import pandas as pd

from DataProcessing import NormalizeTextColumn


def main():
    SampleFrame = pd.DataFrame(
        {
            "StaffId": ["E001"],
            "TalentStatement": ["  I   aspire to  lead product.  "],
        }
    )
    NormalizedFrame = NormalizeTextColumn(SampleFrame)
    print(NormalizedFrame.to_dict(orient="records"))


if __name__ == "__main__":
    main()
