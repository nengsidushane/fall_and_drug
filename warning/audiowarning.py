import subprocess

def audiowarningfall():
    subprocess.Popen('aplay -D hw:3,0 ./warning/fall.wav', shell=True) #语音播报

def audiowarningmedicine():
    subprocess.Popen('aplay -D hw:3,0 ./warning/medicine.wav', shell=True) #语音播报
