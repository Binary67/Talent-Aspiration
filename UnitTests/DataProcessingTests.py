import pandas as pd
import pytest

from DataProcessing import NormalizeTextColumn


def test_normalization_basic():
    InputFrame = pd.DataFrame(
        {
            "StaffId": ["E001"],
            "TalentStatement": ["  I   aspire to  lead product.  "],
        }
    )
    OriginalFrame = InputFrame.copy()
    NormalizedFrame = NormalizeTextColumn(InputFrame)
    ExpectedText = "I aspire to lead product."
    assert NormalizedFrame.loc[0, "TalentStatement"] == ExpectedText
    assert InputFrame.equals(OriginalFrame)
    NormalizedAgain = NormalizeTextColumn(NormalizedFrame)
    assert NormalizedAgain.equals(NormalizedFrame)


def test_missing_column_raises():
    InputFrame = pd.DataFrame({"StaffId": ["E001"]})
    with pytest.raises(KeyError):
        NormalizeTextColumn(InputFrame)


def test_non_string_column_raises():
    InputFrame = pd.DataFrame(
        {"StaffId": ["E001"], "TalentStatement": [123]}
    )
    with pytest.raises(TypeError):
        NormalizeTextColumn(InputFrame)
