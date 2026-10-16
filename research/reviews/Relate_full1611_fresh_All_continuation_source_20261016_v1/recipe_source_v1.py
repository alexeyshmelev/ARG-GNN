"""Read-only command/recipe renderer. Default inert; contains no launch/admit/guard API."""
from pathlib import Path
import hashlib, json
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE=Path(__file__).resolve().parent
RECIPE_SHA = '57f4255e0fe146b846756fc941965ddab9d894f5e5b62f50e4f6b81872d87b43'

def recipe_v1():
    path=HERE/"fresh_All_recipe.inactive.v1.json"
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=RECIPE_SHA:
        raise ValueError("Exact inactive recipe")
    doc=json.loads(raw)
    if doc["status"]!="SOURCE_ONLY_INACTIVE" or doc["execution_authorized"] is not False:
        raise ValueError("This renderer grants no execution")
    return doc

def commands_v1():
    doc=recipe_v1()
    return doc["native_commands"]

if __name__=="__main__":
    print(json.dumps({"status":"SOURCE_ONLY_INACTIVE","execution_authorized":False,
                      "recipe":str(HERE/"fresh_All_recipe.inactive.v1.json"),
                      "native_commands":commands_v1()},sort_keys=True))
