import pandas as Pandas
from DataPreValidation import ValidateInputDataframe


def Main() -> None:
    InputTable = Pandas.DataFrame(
        [
            {
                "StaffId": "S001",
                "TalentStatement": "I want to move into data engineering.",
            },
            {"StaffId": "S002", "TalentStatement": "Happy in current role."},
        ]
    )
    IsValid, Errors = ValidateInputDataframe(InputTable)
    print(f"IsValid: {IsValid}")
    print(f"Errors: {Errors}")


if __name__ == "__main__":
    Main()
