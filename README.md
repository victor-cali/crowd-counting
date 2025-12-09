# 🏖️ Beach Crowd Counting

In summer, beaches are monitored by lifeguards, who gather information about the occupancy, sea state, wind, etc. Image processing techniques can be used to automate this task.

This repository contains the source code and documentation for Project 1 of the course 11761 - Image and Video Analysis course, at the Master's Degree in Intelligent Systems (MUSI), in the University of the Balearic Islands. The aim is to estimate the number of people present on a beach using image processing techniques, a task known as "crowd counting" in Computer Vision. The ultimate goal is to automate the beach monitoring tasks currently performed by lifeguards, such as gathering information about occupancy.

-----

## 🚀 Project Setup Instructions

### 1\. Clone the Repository

### 2\. Environment Setup

Choose your preferred method for setting up the Python virtual environment:

##### **Option A:** Using uv

```bash
# Creates .venv and installs packages from pyproject.toml
uv init
uv pip install --all-features
```

##### **Option B:** Using Traditional `venv` (e.g., for PyCharm)

```bash
# 1. Create the virtual environment (.venv)
python -m venv .venv

# 2. Activate the environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows (Command Prompt or PowerShell):
.venv\Scripts\activate

# 3. Install packages using pip
pip install -r requirements.txt
```

