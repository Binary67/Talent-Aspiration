import pandas as pd
import unicodedata
import re


def NormalizeTextColumn(InputFrame: pd.DataFrame) -> pd.DataFrame:
    """Clean and normalize the TalentStatement text for downstream processing.

    Parameters
    ----------
    InputFrame : pandas.DataFrame
        DataFrame containing StaffId and TalentStatement columns.

    Returns
    -------
    pandas.DataFrame
        New DataFrame with normalized TalentStatement column.
    """
    RequiredColumns = {"StaffId", "TalentStatement"}
    if not RequiredColumns.issubset(InputFrame.columns):
        MissingColumns = RequiredColumns - set(InputFrame.columns)
        MissingList = ", ".join(sorted(MissingColumns))
        raise KeyError(f"InputFrame missing columns: {MissingList}")

    if not pd.api.types.is_string_dtype(InputFrame["TalentStatement"]):
        raise TypeError("TalentStatement column must be string-like")

    NormalizedFrame = InputFrame.copy()
    TalentColumn = NormalizedFrame["TalentStatement"]
    NormalizedFrame["TalentStatement"] = TalentColumn.apply(
        lambda Text: re.sub(
            r"\s+",
            " ",
            unicodedata.normalize("NFKC", Text).strip(),
        )
    )

    return NormalizedFrame
