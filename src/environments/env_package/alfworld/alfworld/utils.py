import os

def mkdirs(dirpath: str) -> str:
    
    try:
        os.makedirs(dirpath)
    except FileExistsError:
        pass

    return dirpath
