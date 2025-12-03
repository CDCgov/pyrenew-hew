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
    A class to manage and execute Julia models.

    This class provides a Python interface for running Julia scripts with proper
    project environment setup and argument handling.

    Attributes
    ----------
    data_json_path : Path
        Absolute path to the input JSON data file.
    model_run_path : Path
        Absolute path to the directory where model outputs will be saved.
    project_path : Path
        Absolute path to the Julia project directory (containing Project.toml).
    julia_entrypoint : Path
        Absolute path to the Julia script that serves as the model entry point.
    model_name : str
        Name identifier for the model.
    nthreads : int
        Number of threads to use for Julia execution (default: 1).

    Methods
    -------
    run(params: dict)
        Execute the Julia model with the specified parameters.
    run_model(*args)
        Low-level method that constructs and runs the Julia command.
    """

    def __init__(
        self,
        data_json_path: str | Path,
        model_run_dir: str | Path,
        model_name: str,
        julia_project_path: str | Path,
        julia_entrypoint: str | Path,
        nthreads: int = 1,
    ):
        """
        Initialize a JuliaModel instance.

        Parameters
        ----------
        data_json_path : str | Path
            Path to the input JSON data file for the model.
        model_run_dir : str | Path
            Directory where model outputs will be saved.
        model_name : str
            Name identifier for the model.
        julia_project_path : str | Path
            Path to the Julia project directory (containing Project.toml).
        julia_entrypoint : str | Path
            Path to the Julia script to execute.
        nthreads : int, optional
            Number of threads for Julia execution (default: 1).
        """
        self.data_json_path = to_abs_path(data_json_path)
        self.model_run_path = to_abs_path(model_run_dir)
        self.project_path = to_abs_path(julia_project_path)
        self.julia_entrypoint = to_abs_path(julia_entrypoint)
        self.model_name = model_name
        self.nthreads = nthreads

    def run(
        self,
        params: dict,
    ):
        """
        Execute the Julia model with specified parameters.

        This method automatically includes the data input path and output directory
        in the parameters before running the model.

        Parameters
        ----------
        params : dict
            Dictionary of command-line arguments to pass to the Julia script.
            Keys will be converted to --key=value format.
            The 'json-input' and 'output-dir' keys are automatically set.

        Raises
        ------
        RuntimeError
            If the Julia model execution fails (non-zero exit code).
        """
        params["json-input"] = self.data_json_path
        params["output-dir"] = self.model_run_path
        args = dict_to_arg_list(params)
        self.run_model(*args)

    def run_model(self, *args):
        """
        Execute the Julia model with low-level command construction.

        This method handles Julia package instantiation, dependency resolution,
        and model execution. It first ensures all Julia dependencies are installed,
        then runs the model script.

        Parameters
        ----------
        *args : str
            Variable-length argument list of command-line arguments to pass to
            the Julia script (typically in --key=value format).

        Raises
        ------
        RuntimeError
            If Julia package instantiation or model execution fails.
        """
        cmd = [
            "julia",
            f"--project={self.project_path}",
            f"--threads={self.nthreads}",
            f"{self.julia_entrypoint}",
            *args,
        ]
        cmd_str = " ".join(cmd)

        print(f"Instantiating Julia model {self.model_name}")
        result = subprocess.run(
            f"julia --project={self.project_path} -e 'using Pkg; Pkg.resolve(); Pkg.instantiate()' ",
            shell=True,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Julia package instantiation failed:\n"
                f"STDOUT: {result.stdout}\n"
                f"STDERR: {result.stderr}"
            )

        print(f"Running experiment with command: {cmd_str}")
        result = subprocess.run(cmd_str, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"Julia model execution failed:\n"
                f"STDOUT: {result.stdout}\n"
                f"STDERR: {result.stderr}"
            )

        print(f"Model {self.model_name} completed successfully")
