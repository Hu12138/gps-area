import json
def getData(path:str):
    data = json.load(open(path))
    return data["points"]