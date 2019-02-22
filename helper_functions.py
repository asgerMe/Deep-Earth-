
import os

def check_path(path):
    is_valid = os.path.isdir(path)
    if is_valid:
        return path
    else:
        return '../'
