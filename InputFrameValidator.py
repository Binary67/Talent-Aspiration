import pandas as pd
import re


def ValidateInputFrame(InputFrame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(InputFrame, pd.DataFrame):
        raise ValueError("InputFrame must be a pandas DataFrame.")
    if InputFrame.empty:
        raise ValueError("InputFrame must contain at least one row.")
    NormalizedColumns = {
        re.sub(r'[^a-z0-9]', '', Column.lower()): Column
        for Column in InputFrame.columns
    }
    RequiredKeys = {'staffid': 'StaffId', 'talentstatement': 'TalentStatement'}
    Missing = [Key for Key in RequiredKeys if Key not in NormalizedColumns]
    if Missing:
        raise ValueError(
            "InputFrame is missing required columns: "
            "StaffId and TalentStatement."
        )
    ValidatedFrame = InputFrame[
        [NormalizedColumns['staffid'], NormalizedColumns['talentstatement']]
    ].copy()
    ValidatedFrame.columns = ['StaffId', 'TalentStatement']
    ValidatedFrame['StaffId'] = ValidatedFrame['StaffId'].astype(str)
    ValidatedFrame['TalentStatement'] = (
        ValidatedFrame['TalentStatement'].astype(str).str.strip()
    )
    ValidatedFrame = ValidatedFrame[
        ValidatedFrame['TalentStatement'].notna()
        & (ValidatedFrame['TalentStatement'] != '')
    ]
    if ValidatedFrame.empty:
        raise ValueError(
            "InputFrame must contain at least one non-empty TalentStatement."
        )
    return ValidatedFrame
