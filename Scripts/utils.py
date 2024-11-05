import os
import subprocess

def create_No_empty_folders(path, start_No, end_No):
    """
    This function creates indexed folders in a specific directory (path) from a start No° and an end No°
    """
    for i in range(start_No, end_No+1):
        if not os.path.exists(path + str(i)):
            os.makedirs(path + str(i))

def execute_3dslicer(script: str, slicer_path):
    """
    With this function you can run 3Dslicer executing the script of your choice.
    :param script: Enter the script you want to execute.
    :return:
    """
    if slicer_path == "":
        slicer_path = "D:\Programmes\Slicer 5.6.2\Slicer.exe"
    if os.path.exists(slicer_path):
        # Run Slicer with the specified script
        subprocess.run([slicer_path, "--no-splash", "--python-script", script])

