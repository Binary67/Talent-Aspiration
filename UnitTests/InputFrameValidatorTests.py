import os as Os
import sys as Sys
import pandas as pd
import pytest

Sys.path.append(Os.path.abspath(Os.path.join(Os.path.dirname(__file__), '..')))
from InputFrameValidator import ValidateInputFrame  # noqa: E402


def test_valid_input_frame():
    InputData = pd.DataFrame({
        'staff_id': ['E001', 'E002'],
        'talent_statement': ['Data scientist', '']
    })
    Result = ValidateInputFrame(InputData)
    assert list(Result.columns) == ['StaffId', 'TalentStatement']
    assert Result.shape[0] == 1
    assert Result.iloc[0]['StaffId'] == 'E001'
    assert Result.iloc[0]['TalentStatement'] == 'Data scientist'


def test_missing_columns():
    InputData = pd.DataFrame({
        'staff_id': ['E001']
    })
    with pytest.raises(ValueError):
        ValidateInputFrame(InputData)


def test_empty_dataframe():
    InputData = pd.DataFrame(columns=['staff_id', 'talent_statement'])
    with pytest.raises(ValueError):
        ValidateInputFrame(InputData)


def test_all_empty_talent_statements():
    InputData = pd.DataFrame({
        'staff_id': ['E001'],
        'talent_statement': ['   ']
    })
    with pytest.raises(ValueError):
        ValidateInputFrame(InputData)
