import json
import time
import pandas as Pandas
from typing import List, Tuple


def ValidateInputDataframe(
    InputTable: Pandas.DataFrame,
) -> Tuple[bool, List[str]]:
    """Validate that the input table meets schema and content requirements."""
    Errors: List[str] = []

    if not isinstance(InputTable, Pandas.DataFrame):
        Errors.append("InputTable must be a pandas DataFrame.")
        return False, Errors

    RowCount = InputTable.shape[0]
    if RowCount == 0:
        Errors.append("InputTable is empty.")
    if RowCount > 1_000_000:
        Errors.append("InputTable exceeds maximum row count of 1,000,000.")

    RequiredColumns = {"StaffId", "TalentStatement"}
    MissingColumns = RequiredColumns - set(InputTable.columns)
    for Column in MissingColumns:
        Errors.append(f"Missing required column: {Column}")

    if not MissingColumns:
        StaffIdSeries = InputTable["StaffId"]
        TalentStatementSeries = InputTable["TalentStatement"]

        if StaffIdSeries.isna().any() or not StaffIdSeries.map(
            lambda Value: isinstance(Value, str)
        ).all():
            Errors.append("Column StaffId must contain strings without nulls.")

        if TalentStatementSeries.isna().any() or not TalentStatementSeries.map(
            lambda Value: isinstance(Value, str)
        ).all():
            Errors.append(
                "Column TalentStatement must contain strings without nulls."
            )
        else:
            InvalidRows = TalentStatementSeries[
                TalentStatementSeries.apply(lambda Value: Value.strip() == "")
            ].index.tolist()
            if InvalidRows:
                Errors.append(
                    "TalentStatement contains empty or whitespace values at "
                    f"rows: {InvalidRows}"
                )

    IsValid = len(Errors) == 0
    return IsValid, Errors


def ValidateJobFunctionsList(
    JobFunctionsList: List[str],
) -> Tuple[bool, List[str], List[str]]:
    """Validate and normalize a list of job functions."""
    Errors: List[str] = []

    if not isinstance(JobFunctionsList, list):
        Errors.append("JobFunctionsList must be a list of strings.")
        return False, [], Errors

    ItemCount = len(JobFunctionsList)
    if ItemCount == 0 or ItemCount > 200:
        Errors.append("JobFunctionsList must contain between 1 and 200 items.")

    NormalizedJobFunctions: List[str] = []
    Seen: set[str] = set()

    for Index, JobFunction in enumerate(JobFunctionsList):
        if not isinstance(JobFunction, str):
            Errors.append(
                f"Job function at index {Index} must be a non-empty string."
            )
            continue

        Trimmed = JobFunction.strip()
        if Trimmed == "":
            Errors.append(
                f"Job function at index {Index} must be a non-empty string."
            )
            continue

        if len(Trimmed) > 60:
            Errors.append(
                f"Job function at index {Index} exceeds 60 characters."
            )
            continue

        LowerName = Trimmed.casefold()
        if LowerName in Seen:
            continue

        Seen.add(LowerName)
        NormalizedJobFunctions.append(Trimmed.title())

    IsValid = len(Errors) == 0
    return IsValid, NormalizedJobFunctions, Errors


def PreflightModels(
    GptConfig: dict,
    EmbeddingModelName: str,
) -> Tuple[bool, bool, List[str]]:
    """Verify GPT connectivity and embedding model readiness."""

    Errors: List[str] = []
    GptReady = False
    EmbeddingModelReady = False

    if not isinstance(GptConfig, dict):
        Errors.append("GptConfig must be a dictionary.")
    else:
        RequiredKeys = {"ApiKey", "ModelName", "MaxRequestsPerMinute"}
        MissingKeys = RequiredKeys - set(GptConfig.keys())
        for Key in MissingKeys:
            Errors.append(f"Missing GptConfig property: {Key}")

        ApiKey = GptConfig.get("ApiKey")
        ModelName = GptConfig.get("ModelName")
        MaxRequests = GptConfig.get("MaxRequestsPerMinute")

        if not isinstance(ApiKey, str):
            Errors.append("GptConfig.ApiKey must be a string.")
        if not isinstance(ModelName, str):
            Errors.append("GptConfig.ModelName must be a string.")
        if not isinstance(MaxRequests, int) or MaxRequests <= 0:
            Errors.append(
                "GptConfig.MaxRequestsPerMinute must be a positive integer."
            )

    if not isinstance(EmbeddingModelName, str) or (
        EmbeddingModelName.strip() == ""
    ):
        Errors.append("EmbeddingModelName must be a non-empty string.")

    if Errors:
        return GptReady, EmbeddingModelReady, Errors

    try:
        import openai

        StartTime = time.time()
        openai.api_key = ApiKey
        Response = openai.ChatCompletion.create(
            model=ModelName,
            messages=[
                {
                    "role": "system",
                    "content": "Echo the user-provided JSON input.",
                },
                {"role": "user", "content": json.dumps({"Ping": "Pong"})},
            ],
            temperature=0,
            max_tokens=50,
            timeout=5,
        )
        Content = Response["choices"][0]["message"]["content"]
        Data = json.loads(Content)
        GptReady = Data.get("Ping") == "Pong"
        Duration = time.time() - StartTime
        if Duration > 5:
            Errors.append(
                "Gpt connectivity test exceeded 5 second warm-up limit."
            )
            GptReady = False
    except Exception as Exc:
        Errors.append(f"Gpt connectivity failed: {Exc}")

    try:
        StartTime = time.time()
        from sentence_transformers import SentenceTransformer

        Model = SentenceTransformer(EmbeddingModelName)
        _ = Model.encode("Probe String")
        EmbeddingModelReady = True
        Duration = time.time() - StartTime
        if Duration > 5:
            Errors.append(
                "Embedding model load exceeded 5 second warm-up limit."
            )
            EmbeddingModelReady = False
    except Exception as Exc:
        Errors.append(f"Embedding model failed: {Exc}")

    return GptReady, EmbeddingModelReady, Errors
