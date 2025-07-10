import subprocess
import time

subprocess.run(["gpio", "mode", "0", "out"])
subprocess.run(["gpio", "mode", "2", "out"])


while True:
    subprocess.run(["gpio", "write", "0", "0"])
    subprocess.run(["gpio", "write", "2", "1"])

    time.sleep(0.5)

    subprocess.run(["gpio", "write", "0", "1"])
    subprocess.run(["gpio", "write", "2", "0"])

    time.sleep(0.5)

subprocess.run(["gpio", "write", "0", "0"])
subprocess.run(["gpio", "write", "2", "0"])
