import os
import sys
import unittest
import pandas as Pandas

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from DataPreValidation import ValidateInputDataframe  # noqa: E402


class DataPreValidationTests(unittest.TestCase):
    def TestValidInputDataframe(self) -> None:
        InputTable = Pandas.DataFrame(
            [
                {
                    "StaffId": "S001",
                    "TalentStatement": (
                        "I want to move into data engineering."
                    ),
                },
                {
                    "StaffId": "S002",
                    "TalentStatement": "Happy in current role.",
                },
            ]
        )
        IsValid, Errors = ValidateInputDataframe(InputTable)
        self.assertTrue(IsValid)
        self.assertEqual(Errors, [])

    def TestMissingRequiredColumn(self) -> None:
        InputTable = Pandas.DataFrame([{"StaffId": "S001"}])
        IsValid, Errors = ValidateInputDataframe(InputTable)
        self.assertFalse(IsValid)
        self.assertIn("Missing required column: TalentStatement", Errors)

    def TestEmptyTalentStatement(self) -> None:
        InputTable = Pandas.DataFrame(
            [{"StaffId": "S001", "TalentStatement": "   "}]
        )
        IsValid, Errors = ValidateInputDataframe(InputTable)
        self.assertFalse(IsValid)
        self.assertTrue(
            any(
                "TalentStatement contains empty or whitespace" in Error
                for Error in Errors
            )
        )

    def TestRowLimitExceeded(self) -> None:
        InputTable = Pandas.DataFrame(
            {
                "StaffId": ["S001"] * 1_000_001,
                "TalentStatement": ["Valid"] * 1_000_001,
            }
        )
        IsValid, Errors = ValidateInputDataframe(InputTable)
        self.assertFalse(IsValid)
        self.assertIn(
            "InputTable exceeds maximum row count of 1,000,000.",
            Errors,
        )

    def TestStaffIdNotString(self) -> None:
        InputTable = Pandas.DataFrame(
            [{"StaffId": 123, "TalentStatement": "Valid"}]
        )
        IsValid, Errors = ValidateInputDataframe(InputTable)
        self.assertFalse(IsValid)
        self.assertIn(
            "Column StaffId must contain strings without nulls.",
            Errors,
        )


def LoadTests() -> unittest.TestSuite:
    Suite = unittest.TestSuite()
    Suite.addTest(DataPreValidationTests("TestValidInputDataframe"))
    Suite.addTest(DataPreValidationTests("TestMissingRequiredColumn"))
    Suite.addTest(DataPreValidationTests("TestEmptyTalentStatement"))
    Suite.addTest(DataPreValidationTests("TestRowLimitExceeded"))
    Suite.addTest(DataPreValidationTests("TestStaffIdNotString"))
    return Suite


if __name__ == "__main__":
    Runner = unittest.TextTestRunner()
    Runner.run(LoadTests())
