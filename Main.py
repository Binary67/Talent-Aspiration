import os
import pandas as Pandas
from DataPreValidation import (
    PreflightModels,
    ValidateInputDataframe,
    ValidateJobFunctionsList,
)


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

    JobFunctionsList = [
        "data engineering",
        "Product Management",
        "Product management",
    ]
    JfIsValid, NormalizedJobFunctions, JfErrors = ValidateJobFunctionsList(
        JobFunctionsList
    )
    print(f"JfIsValid: {JfIsValid}")
    print(f"NormalizedJobFunctions: {NormalizedJobFunctions}")
    print(f"JfErrors: {JfErrors}")

    GptConfig = {
        "ApiKey": os.environ.get("AZURE_OPENAI_API_KEY", ""),
        "ModelName": os.environ.get(
            "AZURE_OPENAI_MODEL", "ConfiguredGptModel"
        ),
        "MaxRequestsPerMinute": 120,
    }
    GptReady, EmbeddingModelReady, PfErrors = PreflightModels(
        GptConfig, "all-MiniLM-L6-v2"
    )
    print(f"GptReady: {GptReady}")
    print(f"EmbeddingModelReady: {EmbeddingModelReady}")
    print(f"PreflightErrors: {PfErrors}")


if __name__ == "__main__":
    Main()
