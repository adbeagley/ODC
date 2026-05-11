"""
Run this script to set up the environment

Author: Aren Beagley
"""

import subprocess
from pathlib import Path
from platform import system
from codecs import decode

# add a code block to check if mamba is available
# otherwise set to use conda


def main():
    """
    Set up or update environment, should be run from base conda environment.
    """
    env_file, env_name = get_yaml_file()

    if not check_envs(env_name):
        print(f"Creating env: {env_name}")
        cmd = ["mamba", "env", "create", "-f", env_file]
        subprocess.run(cmd, check=True, shell=False)
    else:
        # update env to add all the packages from the yml file
        print("Environment already exists. Updating based on yaml file.")
        cmd = ["mamba", "env", "update", "-f", env_file, "--prune"]
        subprocess.run(cmd, check=True, shell=False)

    cmds = [
        ["conda", "env", "list"],
        ["conda", "run", "-n", env_name, "conda", "list"],
    ]

    for cmd in cmds:
        subprocess.run(cmd, check=True, shell=False)


def uninstall_wheels():
    # activate correct env and then run this to remove the pip packages
    # pip freeze | xargs pip uninstall -y
    pass


def check_envs(env_name: str):
    """Checks for existing environment with name"""
    envs_list = subprocess.run(
        ["conda", "env", "list"], capture_output=True, check=True, shell=False
    )
    # get list of lines as strings
    envs_list = decode(str(envs_list.stdout), "unicode_escape").splitlines()
    for line in envs_list:
        if line.startswith(env_name):
            return True
    return False


def get_yaml_file():
    """Get appropriate yml file for OS"""
    current_os = system()
    wdir = Path(__file__).parent.resolve()

    # if current_os.lower() == "windows":
    #     files_dir = wdir.joinpath("windows")
    # elif current_os.lower() == "linux":
    #     files_dir = wdir.joinpath("linux")
    # elif current_os.lower() == "darwin":
    #     files_dir = wdir.joinpath("mac")
    # else:
    #     raise RuntimeError("Unknown Operating System")

    # ymls = [fpath for fpath in files_dir.glob('*.yaml')]

    ymls = [fpath for fpath in wdir.glob("*.yaml")]

    if len(ymls) == 0:
        raise RuntimeError(f"No yml file for OS: {current_os}")

    if len(ymls) > 1:
        raise RuntimeError("Multiple yml files detected.")

    with open(ymls[0], encoding="utf8") as file:
        first_line = file.readline().strip("\n")
        env_name = first_line.split(":")[-1].strip(" ")

    return ymls[0], env_name


if __name__ == "__main__":
    main()
