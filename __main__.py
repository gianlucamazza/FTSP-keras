import subprocess


def run_script(script_name):
    try:
        subprocess.run(["python", script_name], check=True)
        print(f"Script {script_name} executed successfully.")
    except subprocess.CalledProcessError:
        print(f"An error occurred while executing {script_name}.")


if __name__ == "__main__":
    scripts = [
        "data_preparation.py",
        "feature_engineering.py",
        "model.py",
        "train.py",
        "predict.py"
    ]

    for script in scripts:
        run_script(script)
