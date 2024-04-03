import subprocess

scripts = [
    "create_face_datasets.py",
    "training_model.py",
    "lock_unlock_face_recognition.py",
    "utils.py",
    "mouse-cursor-control.py"
]

for script in scripts:
    try:
        subprocess.run(["python", script], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running {script}: {e}")

