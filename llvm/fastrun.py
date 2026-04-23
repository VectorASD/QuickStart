import not_torch   # type: ignore

try: import sqlalchemy
except ImportError: exit("sudo pip install sqlalchemy")



import os, sys
sys.path.insert(0, os.path.join(os.path.expanduser("~"), "FlagGems", "src")) # ничё не знаю, pytest также делает)))

class Dummy:
    @staticmethod
    def getoption(name):
        return

import importlib
conftest = importlib.import_module("tests.conftest")
conftest.pytest_configure(Dummy)
print("Loaded conftest")



import torch

if __name__ == "__main__":
    tests = importlib.import_module("tests.test_tensor_constructor_ops")
    tests.test_accuracy_eye((2, 2), dtype=torch.bfloat16)
    # result = eye_m(2, 2, dtype=torch.bfloat16)
    # print(result)
