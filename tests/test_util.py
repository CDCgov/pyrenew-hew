import pytest

from pyrenew_hew.util import (
    hew_letters_from_flags,
    pyrenew_model_name_from_flags,
)


@pytest.mark.parametrize(
    [
        "fit_ed_visits",
        "fit_hospital_admissions",
        "fit_wastewater",
        "expected_letters",
    ],
    [
        (False, False, False, "null"),
        (False, True, False, "h"),
        (True, False, False, "e"),
        (False, False, True, "w"),
        (True, True, False, "he"),
        (False, True, True, "hw"),
        (True, False, True, "ew"),
        (True, True, True, "hew"),
    ],
)
def test_hew_naming_from_flags(
    fit_ed_visits, fit_hospital_admissions, fit_wastewater, expected_letters
):
    assert (
        hew_letters_from_flags(
            fit_ed_visits, fit_hospital_admissions, fit_wastewater
        )
        == expected_letters
    )

    assert (
        pyrenew_model_name_from_flags(
            fit_ed_visits, fit_hospital_admissions, fit_wastewater
        )
        == f"pyrenew_{expected_letters}"
    )
