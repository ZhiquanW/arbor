import os, sys


sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../src"))

import stgui.starbor as starbor  # noqa: E402

if __name__ == "__main__":
    starbor_app = starbor.Starbor()
    starbor_app.launch()
