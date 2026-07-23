

import os

cmbagent_debug = os.getenv("CMBAGENT_DEBUG", "False").lower() == "true"
cmbagent_disable_display = os.getenv("CMBAGENT_DISABLE_DISPLAY", "True").lower() == "true"
streamlit_on = os.getenv("STREAMLIT_ON", "False").lower() == "true"


# see https://github.com/openai/openai-python/blob/da48e4cac78d1d4ac749e2aa5cfd619fde1e6c68/src/openai/types/beta/file_search_tool.py#L20
# default_file_search_max_num_results = 20
# The default is 20 for `gpt-4*` models and 5 for `gpt-3.5-turbo`. This number
# should be between 1 and 50 inclusive.
file_search_max_num_results = 20


# Define the color mapping
cmbagent_color_dict = {
    "admin": "green",
    "control": "red"
}
cmbagent_default_color = "yellow"



# Define the logo as a module-level constant.
# LOGO = r"""
# """

LOGO = r"""
Multi-Agent Systems for Autonomous Discovery


Get the source code [here](https://github.com/CMBAgents/cmbagent/tree/main).

Built with [AG2](https://github.com/ag2ai/ag2).
"""


# Define the logo as a module-level constant.
# LOGO = r"""
#  _____ ___  _________  ___  _____  _____ _   _ _____ 
# /  __ \|  \/  || ___ \/ _ \|  __ \|  ___| \ | |_   _|
# | /  \/| .  . || |_/ / /_\ \ |  \/| |__ |  \| | | |  
# | |    | |\/| || ___ \  _  | | __ |  __|| . ` | | |  
# | \__/\| |  | || |_/ / | | | |_\ \| |___| |\  | | |  
# \_____/\_|  |_/\____/\_| |_/\____/\____/\_| \_/ \_/  
#     multi-agent systems for autonomous discovery    

# Built with AG2
# Version: Beta3
# Last updated: 11/03/5202
# """

# Calculate the image width as a module-level variable.
_lines = LOGO.splitlines()
_ascii_width = max(len(line) for line in _lines)
_scaling_factor = 4  # For example, 8 pixels per character.
IMG_WIDTH = _ascii_width * _scaling_factor

cmbagent_gui_mode = os.getenv("CMBAGENT_GUI_MODE", "False").lower() == "true" ## not used 
cmbagent_gui_mode = True
# print("\n in cmbagent_utils.py cmbagent_gui_mode: ", cmbagent_gui_mode)
# import sys; sys.exit()
