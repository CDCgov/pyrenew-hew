from typing import Iterable

import pytest

from pyrenew_hew.util import (
    hew_letters_from_flags,
    hew_models,
    powerset,
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
