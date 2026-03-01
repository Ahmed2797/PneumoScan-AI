from pathlib import Path

def get_project_root() -> Path:
    """
    Locate the project root directory by searching for a marker file.

    Traverses up from the current file's directory until it finds a 
    'setup.py' file. If no marker is found, it defaults to the 
    current working directory.

    Returns:
        Path: The absolute path to the project root directory.
    """
    path = Path(__file__).resolve()
    for parent in path.parents:
        # Detect project root by checking for setup.py
        if (parent / "setup.py").exists():
            return parent
    return Path.cwd()

# Define the root directory for the project
ROOT_DIR = get_project_root()

# Directory containing configuration files
YAML_DIR = ROOT_DIR / "yamlfile"

# Absolute paths to specific YAML configuration files
CONFIG_YAML_FILE = YAML_DIR / "config.yaml"
"""Path: Configuration settings for the application environment."""

PARAM_YAML_FILE  = YAML_DIR / "param.yaml"
"""Path: Hyperparameters and model constants."""

SECRET_YAML_FILE = ROOT_DIR / "yamlfile" / "secrets.yaml"
"""Path: Sensitive credentials (API keys, DB passwords). Ensure this is in .gitignore!"""