import argparse
import logging
import os
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

import numpyro
import polars as pl
from prep_data import process_and_save_state

numpyro.set_host_device_count(4)

from fit_model import fit_and_save_model  # noqa
from generate_predictive import generate_and_save_predictions  # noqa
