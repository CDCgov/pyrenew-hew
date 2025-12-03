import subprocess
from pathlib import Path
from typing import List, Union


def dict_to_arg_list(input_dict: dict) -> List[str]:
    """
    Convert a dictionary into a list of strings with the pattern "--key=value".

    Parameters
    ----------
    input_dict : dict
        The dictionary to be converted. Keys and values should be strings.

    Returns
    -------
    List[str]
        A list of strings in the format "--key=value" for each key-value pair in the dictionary.
    """
    return [f"--{key}={value}" for key, value in input_dict.items()]


def to_abs_path(path: Union[str, Path]) -> Path:
    """
    Convert a relative or string path to an absolute Path object.

    This function ensures that the input path is converted to a `Path` object
    and resolved to its absolute form.

    Parameters
    ----------
    path : Union[str, Path]
        The input path, which can be a string or a `Path` object.

    Returns
    -------
    Path
        The absolute path as a `Path` object.
    """
    if isinstance(path, str):
        path = Path(path)
    return path.resolve()


class JuliaModel:
    """
    A class to manage and execute experiments using a Julia model.

    Attributes:
        experiment_path (str): Absolute path to the experiment directory.
        model_name (str): Name of the model used in the experiment.
        target (str): Target variable for the experiment, derived from the data source.
        train_data (Any): Training data used in the experiment.
        eval_data (Any): Evaluation data used in the experiment.
        reference_date (Any): Reference date for the experiment.
        nthreads (int): Number of threads to use during model execution.
        model_run_path (str): Path to the Julia script or executable to run the model.
        project_path (str): Path to the Julia project directory.

    Methods:
        __init__(experiment_info, model_name, model_run_path, project_path, nthreads):
            Initializes the JuliaExperiment instance with experiment configuration,
            model name, model run path, and project path.

        run(location, params):
            Executes the experiment by logging model details, running the model, and logging artifacts to MLflow.

        log_model(params, location):
            Logs experiment parameters and tags to MLflow.

        log_artifact():
            Logs experiment artifacts (plots and forecast data) to MLflow.

        score_model():
            Logs forecast scores to MLflow.

        get_tags():
            Returns a dictionary of tags associated with the experiment, including the model name.

        run_model(location, *args):
            Executes the Julia model with the specified location and additional arguments.
            Handles Julia package instantiation and resolves dependencies before running the experiment.
    """

    def __init__(
        self,
        data_json_path: str | Path,
        model_run_dir: str | Path,
        model_name: str,
        julia_project_path: str | Path,
        nthreads: int = 1,
    ):
        self.data_json_path = to_abs_path(data_json_path)
        self.model_run_path = to_abs_path(model_run_dir)
        self.project_path = to_abs_path(julia_project_path)
        self.model_name = model_name
        self.nthreads = nthreads

    def run(
        self,
        params: dict,
    ):
        params["threads"] = self.nthreads
        params["project"] = self.project_path
        params["json-input"] = self.data_json_path
        params["output-dir"] = self.model_run_path
        args = dict_to_arg_list(params)
        self.run_model(*args)

    def run_model(self, *args):
        cmd = [
            "julia",
            *args,
        ]
        cmd_str = " ".join(cmd)
        print(f"instantiate Julia model {self.model_name}")
        subprocess.run(
            f"julia --project={self.project_path} -e 'using Pkg; Pkg.resolve(); Pkg.instantiate()' ",
            shell=True,
        )
        print(f"run experiment with command: {cmd_str}")
        subprocess.run(cmd_str, shell=True)
