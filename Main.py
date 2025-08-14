import pandas as pd
from InputFrameValidator import ValidateInputFrame


def Run():
    SampleData = pd.DataFrame({
        'staff_id': ['E001', 'E002', 'E003'],
        'talent_statement': ['I like data.', ' ', 'Lead developer']
    })
    ValidatedFrame = ValidateInputFrame(SampleData)
    print(ValidatedFrame)


if __name__ == '__main__':
    Run()
