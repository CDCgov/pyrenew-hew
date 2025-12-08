from collections.abc import Iterable

import pytest

from pyrenew_hew.utils import (
    flags_from_hew_letters,
    flags_from_pyrenew_model_name,
    hew_letters_from_flags,
    hew_models,
    powerset,
    pyrenew_model_name_from_flags,
    validate_hew_letters,
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
    expected_model_name = f"pyrenew_{expected_letters}"
    assert (
        hew_letters_from_flags(fit_ed_visits, fit_hospital_admissions, fit_wastewater)
        == expected_letters
    )

    assert (
        pyrenew_model_name_from_flags(
            fit_ed_visits, fit_hospital_admissions, fit_wastewater
        )
        == expected_model_name
    )

    assert flags_from_hew_letters(expected_letters) == dict(
        fit_ed_visits=fit_ed_visits,
        fit_hospital_admissions=fit_hospital_admissions,
        fit_wastewater=fit_wastewater,
    )
    assert flags_from_hew_letters(expected_letters, flag_prefix="other_prefix") == dict(
        other_prefix_ed_visits=fit_ed_visits,
        other_prefix_hospital_admissions=fit_hospital_admissions,
        other_prefix_wastewater=fit_wastewater,
    )

    assert flags_from_pyrenew_model_name(expected_model_name) == dict(
        fit_ed_visits=fit_ed_visits,
        fit_hospital_admissions=fit_hospital_admissions,
        fit_wastewater=fit_wastewater,
    )


def test_validate_hew_letters():
    with pytest.raises(ValueError, match="Expected either a string"):
        validate_hew_letters("hewr")
    with pytest.raises(ValueError, match="Expected either a string"):
        validate_hew_letters("nulle")
    validate_hew_letters("hew")
    validate_hew_letters("heewwh")


def test_flag_from_string_errors():
    with pytest.raises(ValueError, match="Expected either a string"):
        flags_from_hew_letters("hewr")
    with pytest.raises(ValueError, match="Expected either a string"):
        flags_from_hew_letters("nulle")
    with pytest.raises(ValueError, match="pyrenew_"):
        flags_from_pyrenew_model_name("a_pyrenew_hew")


@pytest.mark.parametrize(
    "test_items",
    [
        range(10),
        ["a", "b", "c"],
        [None, "a", "b"],
        [None, None, "a", "b"],
        ["a", "b", "a", "a"],
        [1, 1, 1.5, 2],
    ],
)
def test_powerset(test_items):
    pset_iter = powerset(test_items)
    pset = set(pset_iter)
    assert isinstance(pset_iter, Iterable)
    assert set([(item,) for item in test_items]).issubset(pset)
    assert len(pset) == 2 ** len(set(test_items))
    assert () in pset


def test_hew_model_iterator():
    expected = [
        (),
        ("h",),
        ("e",),
        ("w",),
        (
            "e",
            "w",
        ),
        (
            "h",
            "e",
        ),
        (
            "h",
            "w",
        ),
        ("h", "e", "w"),
    ]
    assert set([tuple(sorted(i)) for i in hew_models()]) == set(
        [tuple(sorted(i)) for i in expected]
    )
    assert set([tuple(sorted(i)) for i in hew_models(False)]) == set(
        [tuple(sorted(i)) for i in filter(None, expected)]
    )
