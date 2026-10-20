"""Select finite likelihood observations without changing the latent time grid."""

import numpy as np
from pyrenew_multisignal.hew import PyrenewHEWData


class PyRenewObservationData(PyrenewHEWData):
    """Retain source dates for model duration, but omit missing likelihood terms.

    The upstream data class passes nulls through as NaNs to NumPyro. Selecting
    both observations and their calendar indices avoids NaN likelihoods without
    imputing counts or moving the right-truncation CDF when the tail is missing.
    """

    @staticmethod
    def _observed(values):
        mask = np.isfinite(values)
        if not mask.any():
            raise ValueError(
                "Each fitted signal requires at least one finite observation"
            )
        return mask

    @property
    def data_observed_disease_ed_visits(self):
        values = super().data_observed_disease_ed_visits
        return None if values is None else values[self._observed(values)]

    @property
    def model_t_obs_ed_visits(self):
        indices = super().model_t_obs_ed_visits
        values = super().data_observed_disease_ed_visits
        return None if values is None else indices[self._observed(values)]

    @property
    def data_observed_disease_hospital_admissions(self):
        values = super().data_observed_disease_hospital_admissions
        return None if values is None else values[self._observed(values)]

    @property
    def model_t_obs_hospital_admissions(self):
        indices = super().model_t_obs_hospital_admissions
        values = super().data_observed_disease_hospital_admissions
        return None if values is None else indices[self._observed(values)]
