import pandas as pd
import pytest

import bib_dedupe.conditions
from bib_dedupe.constants import Fields


def test_conditions() -> None:
    df = pd.DataFrame(
        columns=[
            Fields.AUTHOR,
            Fields.TITLE,
            Fields.VOLUME,
            Fields.PAGES,
            Fields.ABSTRACT,
            Fields.ISBN,
            Fields.CONTAINER_TITLE,
            Fields.NUMBER,
            Fields.DOI,
            Fields.YEAR,
            Fields.ENTRYTYPE,
            "title_partial_ratio",
            "container_title_1",
            "container_title_2",
            "doi_1",
            "doi_2",
            "volume_1",
            "volume_2",
            "title_1",
            "title_2",
            "ENTRYTYPE_1",
            "ENTRYTYPE_2",
            "number_1",
            "number_2",
            "pages_1",
            "pages_2",
            "author_1",
            "author_2",
            "abstract_1",
            "abstract_2",
            "year_1",
            "year_2",
        ]
    )

    for condition in (
        bib_dedupe.conditions.updated_pair_conditions
        + bib_dedupe.conditions.non_duplicate_conditions
        + bib_dedupe.conditions.duplicate_conditions
    ):
        try:
            print(condition)
            df.query(condition)
        except Exception as e:
            pytest.fail(f"Condition '{condition}' could not be parsed. Error: {str(e)}")