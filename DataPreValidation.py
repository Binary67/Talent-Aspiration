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
