import os
import sys
import types
import unittest
from unittest.mock import MagicMock, patch
import pandas as Pandas

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from DataPreValidation import (  # noqa: E402
    ValidateInputDataframe,
    ValidateJobFunctionsList,
    PreflightModels,
)


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

    def TestValidJobFunctionsList(self) -> None:
        JobFunctionsList = [
            "data engineering",
            "Product Management",
            "Product management",
        ]
        IsValid, NormalizedJobFunctions, Errors = ValidateJobFunctionsList(
            JobFunctionsList
        )
        self.assertTrue(IsValid)
        self.assertEqual(
            NormalizedJobFunctions, ["Data Engineering", "Product Management"]
        )
        self.assertEqual(Errors, [])

    def TestJobFunctionsListEmpty(self) -> None:
        IsValid, NormalizedJobFunctions, Errors = ValidateJobFunctionsList([])
        self.assertFalse(IsValid)
        self.assertEqual(NormalizedJobFunctions, [])
        self.assertIn(
            "JobFunctionsList must contain between 1 and 200 items.", Errors
        )

    def TestJobFunctionTooLong(self) -> None:
        LongName = "A" * 61
        IsValid, NormalizedJobFunctions, Errors = ValidateJobFunctionsList(
            [LongName]
        )
        self.assertFalse(IsValid)
        self.assertEqual(NormalizedJobFunctions, [])
        self.assertTrue(
            any("exceeds 60 characters" in Error for Error in Errors)
        )

    def TestJobFunctionNotString(self) -> None:
        IsValid, NormalizedJobFunctions, Errors = ValidateJobFunctionsList(
            ["Valid", 123]
        )
        self.assertFalse(IsValid)
        self.assertEqual(NormalizedJobFunctions, ["Valid"])
        self.assertTrue(
            any("must be a non-empty string" in Error for Error in Errors)
        )

    def TestJobFunctionsListTooMany(self) -> None:
        JobFunctionsList = [f"Role{i}" for i in range(201)]
        IsValid, NormalizedJobFunctions, Errors = ValidateJobFunctionsList(
            JobFunctionsList
        )
        self.assertFalse(IsValid)
        self.assertEqual(len(NormalizedJobFunctions), 201)
        self.assertIn(
            "JobFunctionsList must contain between 1 and 200 items.", Errors
        )

    def TestPreflightModelsSuccess(self) -> None:
        GptConfig = {
            "ApiKey": "Key",
            "ModelName": "Model",
            "MaxRequestsPerMinute": 60,
        }
        MockChat = MagicMock()
        MockChat.create.return_value = {
            "choices": [{"message": {"content": "{\"Ping\": \"Pong\"}"}}]
        }
        MockOpenAi = types.SimpleNamespace(ChatCompletion=MockChat)
        MockModelInstance = MagicMock()
        MockModelInstance.encode.return_value = [0.1]
        MockSentence = MagicMock(return_value=MockModelInstance)
        MockSt = types.SimpleNamespace(SentenceTransformer=MockSentence)
        with patch.dict(
            sys.modules,
            {"openai": MockOpenAi, "sentence_transformers": MockSt},
        ):
            GptReady, EmbeddingModelReady, Errors = PreflightModels(
                GptConfig, "all-MiniLM-L6-v2"
            )
        self.assertTrue(GptReady)
        self.assertTrue(EmbeddingModelReady)
        self.assertEqual(Errors, [])

    def TestPreflightModelsGptFailure(self) -> None:
        GptConfig = {
            "ApiKey": "Key",
            "ModelName": "Model",
            "MaxRequestsPerMinute": 60,
        }
        MockChat = MagicMock()
        MockChat.create.side_effect = Exception("Failure")
        MockOpenAi = types.SimpleNamespace(ChatCompletion=MockChat)
        MockModelInstance = MagicMock()
        MockModelInstance.encode.return_value = [0.1]
        MockSentence = MagicMock(return_value=MockModelInstance)
        MockSt = types.SimpleNamespace(SentenceTransformer=MockSentence)
        with patch.dict(
            sys.modules,
            {"openai": MockOpenAi, "sentence_transformers": MockSt},
        ):
            GptReady, EmbeddingModelReady, Errors = PreflightModels(
                GptConfig, "all-MiniLM-L6-v2"
            )
        self.assertFalse(GptReady)
        self.assertTrue(EmbeddingModelReady)
        self.assertTrue(
            any("Gpt connectivity failed" in Error for Error in Errors)
        )

    def TestPreflightModelsEmbeddingFailure(self) -> None:
        GptConfig = {
            "ApiKey": "Key",
            "ModelName": "Model",
            "MaxRequestsPerMinute": 60,
        }
        MockChat = MagicMock()
        MockChat.create.return_value = {
            "choices": [{"message": {"content": "{\"Ping\": \"Pong\"}"}}]
        }
        MockOpenAi = types.SimpleNamespace(ChatCompletion=MockChat)
        MockSentence = MagicMock(side_effect=Exception("LoadError"))
        MockSt = types.SimpleNamespace(SentenceTransformer=MockSentence)
        with patch.dict(
            sys.modules,
            {"openai": MockOpenAi, "sentence_transformers": MockSt},
        ):
            GptReady, EmbeddingModelReady, Errors = PreflightModels(
                GptConfig, "all-MiniLM-L6-v2"
            )
        self.assertTrue(GptReady)
        self.assertFalse(EmbeddingModelReady)
        self.assertTrue(
            any("Embedding model failed" in Error for Error in Errors)
        )


def LoadTests() -> unittest.TestSuite:
    Suite = unittest.TestSuite()
    Suite.addTest(DataPreValidationTests("TestValidInputDataframe"))
    Suite.addTest(DataPreValidationTests("TestMissingRequiredColumn"))
    Suite.addTest(DataPreValidationTests("TestEmptyTalentStatement"))
    Suite.addTest(DataPreValidationTests("TestRowLimitExceeded"))
    Suite.addTest(DataPreValidationTests("TestStaffIdNotString"))
    Suite.addTest(DataPreValidationTests("TestValidJobFunctionsList"))
    Suite.addTest(DataPreValidationTests("TestJobFunctionsListEmpty"))
    Suite.addTest(DataPreValidationTests("TestJobFunctionTooLong"))
    Suite.addTest(DataPreValidationTests("TestJobFunctionNotString"))
    Suite.addTest(DataPreValidationTests("TestJobFunctionsListTooMany"))
    Suite.addTest(
        DataPreValidationTests("TestPreflightModelsSuccess")
    )
    Suite.addTest(
        DataPreValidationTests("TestPreflightModelsGptFailure")
    )
    Suite.addTest(
        DataPreValidationTests("TestPreflightModelsEmbeddingFailure")
    )
    return Suite


if __name__ == "__main__":
    Runner = unittest.TextTestRunner()
    Runner.run(LoadTests())
