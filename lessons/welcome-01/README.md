# Set up your learning workspace

Phase 00: Setup & first steps · about 40 minutes · CPU

This reference explicitly selects CPU. Uploading the same file or setting JAX_PLATFORMS does not adapt it to TPU. Use the execution matrix to review device selection and data ownership before adapting a separate target program.

Download the beginner workspace: public/downloads/jax-start-here.zip

In the lesson reader, use the beginner workspace download button. In a local checkout, the bundle is under public/downloads/.

## What you will be able to do

- Open a terminal in the folder containing your experiment.
- Create a project-specific Python environment and check which interpreter you are using.
- Install the course packages and run a saved Python file.
- Recognize a successful result, save the output, and resume in a new terminal.

## The problem

Let’s get your first JAX experiment running. You do not need to know Git or terminal commands yet: we will show where to type each command and how to check the result. You will need a computer, an internet connection for the downloads, and permission to install Python. By the end, you’ll have one experiment and a record of what it produced. A laptop CPU is enough.

## The idea

Your first goal is a small, repeatable result: save a Python file, run it with the intended Python environment, and explain its output. A terminal accepts commands; Python executes the program; JAX is a package that Python imports. Keeping those three roles separate makes setup errors much easier to locate.

## Follow a file from saving to running

Imagine that your editor shows first_experiment.py, but the terminal says the file does not exist. The program has not reached JAX yet. First save the file, check its full filename, and compare its folder with the terminal's working folder. A file visible in an editor can still be unsaved or stored elsewhere.

Now suppose Python finds the file but cannot import JAX. That is a different boundary: the selected interpreter cannot find the package. Check the interpreter path and its environment before installing anything again. On Windows, the commands below select the environment's Python directly; shell activation is not required.

Once the calculation runs, retain the command and printed result together. Tomorrow, the command tells you how to reproduce the result; a screenshot of the number alone cannot identify the file or environment that produced it.

### Where the first result comes from

**Predict:** The notebook imports JAX, but the terminal script does not. What should you compare first?

![Where the first result comes from](../../phases/00-welcome/01-set-up-your-learning-workspace/outputs/mechanism.svg)

*Architecture and dataflow mechanism diagram.*

Read downward: a command runs from a working folder, selects an interpreter, opens the file and imports its packages before computing. Arrows are dependencies, not measured time. A file-path failure happens before JAX arithmetic; an import failure points to the selected environment.

### Pause and reason

The notebook imports JAX, but the terminal script does not. What should you compare first?

<details><summary>Compare your reasoning</summary>

Compare the Python executables used by the notebook kernel and terminal. They can belong to different environments. Install into, or select, the intended environment only after identifying that difference.

</details>

## 1. Download and unpack your workspace

Use the “Download the beginner workspace” button above. Open your Downloads folder and unzip jax-start-here.zip. On Windows, right-click it and choose Extract All; on macOS, double-click it. Open the extracted jax-start-here folder. Do not work inside the ZIP preview.

You should see first_experiment.py, requirements-beginner.txt, and START-HERE.md. first_experiment.py is the program you will run. requirements-beginner.txt lists the package versions to install. START-HERE.md is a copy of these instructions. Keep your own files alongside them, outside .venv. You do not need to clone the repository for this lesson.

## 2. Check that Python is installed

Open a terminal. On macOS, press Command–Space, type Terminal, and press Return. On Windows, open Start, type PowerShell, and open it. On Linux, open your desktop’s Terminal application.

Choose the command for your operating system below. Type or paste one line, then press Enter. Do not type a prompt symbol such as $ or PS>. If you see >>>, you are already inside the Python interpreter; type `exit()` and press Enter to return to the terminal before using these commands.

The course setup supports Python >=3.11,<3.15. The CPU example was checked with Python 3.14.3, and the commands below use Python 3.14. If Python 3.14 is unavailable, choose another supported release in that range and replace `3.14` in the commands below with its major and minor version, such as `3.12`. Open the official Python downloads link in the references, select a supported installer for macOS or Windows, and follow the installer. On Windows, the launcher or install manager supplies `py`; on macOS, the Python installer supplies `python3.14`. Close and reopen the terminal after installing, then retry the version check. Linux installation depends on the distribution: use its matching Python installation instructions and return here once the version check works. Do not continue with a Python version outside the supported range.

**macOS / Linux — terminal**

```sh
# Run macos / linux — terminal using the course Python environment
python3.14 --version
```

**Expected:** A line beginning Python 3.14. The patch number may differ from the tested 3.14.3.

**Windows — PowerShell**

```powershell
# Windows — PowerShell
py -3.14 --version
```

**Expected:** A line beginning Python 3.14. If `py` cannot find it, install Python 3.14 first.

## 3. Put the terminal in the right folder

The program and requirements file are in the extracted workspace. A command using a filename looks for it relative to the terminal’s current folder. We need the terminal to be in jax-start-here.

macOS: type cd followed by a space, drag the extracted jax-start-here folder from Finder into the terminal, then press Enter. Dragging inserts its actual path, including spaces correctly. Linux: use cd with the actual folder path in quotes; your file manager may offer “Open in Terminal.” Windows: open the extracted folder in File Explorer, click its address bar, type powershell, and press Enter. That opens PowerShell in this folder.

Run the appropriate check below. If requirements-beginner.txt is missing, move into the extracted folder containing that file before continuing. “Repository root” in later lessons means the top folder of the full course checkout; this beginner workspace is enough for the current experiment.

**macOS / Linux — check folder and files**

```sh
# macOS / Linux — check folder and files
pwd
ls
```

**Expected:** The path ends with your extracted workspace folder, and the file list includes first_experiment.py and requirements-beginner.txt.

**Windows — check folder and files**

```powershell
# Windows — check folder and files
Get-Location
Get-ChildItem
```

**Expected:** The displayed folder contains first_experiment.py and requirements-beginner.txt.

## 4. Create the environment and install the packages

Run the lines for your operating system in order, staying in the workspace folder. The venv command creates .venv using the Python version you just selected. It often finishes without printing anything. That is normal; wait until the terminal is ready for another command. If you already have a .venv from another Python version in this folder, use a fresh extracted workspace for this walkthrough.

On macOS/Linux, source loads the activation script into this terminal. You may see (.venv) at the start of the prompt. The important check is sys.executable: it should point inside this workspace’s .venv/bin. On Windows we use .venv\Scripts\python.exe explicitly, so no activation script or execution-policy change is needed.

The pip command is the package installer. `-r` means “read the package list from this file.” It downloads and installs packages into the selected environment. Let it finish. Download messages are normal; a traceback or an ERROR means the step did not succeed. The final import check below is more useful than guessing from the download log. The pins in requirements-beginner.txt describe the minimal tested JAX/NumPy CPU setup; accelerator installation is a later lesson.

**macOS / Linux — run one line at a time**

```sh
# Run macos / linux — run one line at a time using the course Python environment
python3.14 -m venv .venv
source .venv/bin/activate
python -c "import sys; print(sys.executable)"
python -m pip install -r requirements-beginner.txt
python -c "import jax, numpy; print(jax.__version__, numpy.__version__)"
```

**Expected:** The interpreter path includes .venv/bin/python. The final line is 0.9.2 2.4.4.

**Windows — run one line at a time**

```powershell
# Windows — run one line at a time
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -c "import sys; print(sys.executable)"
.\.venv\Scripts\python.exe -m pip install -r requirements-beginner.txt
.\.venv\Scripts\python.exe -c "import jax, numpy; print(jax.__version__, numpy.__version__)"
```

**Expected:** The interpreter path includes .venv\Scripts\python.exe. The final line is 0.9.2 2.4.4. These PowerShell instructions have not been executed on Windows in this project.

## 5. Run your first saved program

Open first_experiment.py in a text/code editor if you want to see its contents. The same Python program is shown below in “Run the example.” Python code belongs in that file; the commands below belong in the terminal. The downloaded file is already saved, so you can run it without copying any code.

Run the command for your operating system. The program prints the Python and package versions, the backend, the devices, and a sum. The starter explicitly selects CPU before importing JAX, so the expected backend here is cpu. The device description can vary; the final result should be Sum: $6.0$. The program constructs the numbers $0$, $1$, $2$, $3$; adding them gives $6$. This known result checks that we ran a numerical computation, not only installed a package.

If you see a traceback, read the final line and compare it with the troubleshooting section. Do not count a failed command as a completed experiment.

**macOS / Linux — activated terminal**

```sh
# Run macos / linux — activated terminal using the course Python environment
python first_experiment.py
```

**Expected:** Python: 3.14.x
JAX: 0.9.2
NumPy: 2.4.4
Backend: cpu
Devices: a CPU device description
Sum: 6.0

**Windows — PowerShell**

```powershell
# Run windows — powershell using the course Python environment
.\.venv\Scripts\python.exe first_experiment.py
```

**Expected:** The same version/backend/result lines. CPU device wording may differ.

## 6. Make one change and save your result

In the editor, find `arange(4, ...)` in first_experiment.py and change $4$ to $6$. The array now contains $0$ through $5$. Predict their sum before running. Also change the final assert from `== 6.0` to `== 15.0`, because that assertion is the program’s expected-result check. Save the file with its .py extension; a filename such as first_experiment.py.txt will not match the run command.

Run it again using the Step 5 command. The final line should be Sum: $15.0$. If it still says $6.0$, check that you saved the file and that the terminal is in the same workspace folder you edited.

After that successful run, the commands below run it once more and save its printed output to my-first-run.txt. Open that text file and check that the versions, CPU information, and sum are present. Add a note in your editor explaining the one change and your prediction. The > symbol saves output to a file and replaces its previous contents; choose another filename when you want to keep multiple runs.

**macOS / Linux — save output**

```sh
# Run macos / linux — save output using the course Python environment
python first_experiment.py > my-first-run.txt
```

**Expected:** my-first-run.txt appears in the workspace and ends with Sum: 15.0 after your saved edit.

**Windows — save output**

```powershell
# Run windows — save output using the course Python environment
.\.venv\Scripts\python.exe first_experiment.py > my-first-run.txt
```

**Expected:** my-first-run.txt contains the program output after your saved edit.

## 7. Resume when you open a new terminal

The .venv folder stays on disk when you close the terminal. On macOS/Linux, activation only applied to the terminal you used; a new terminal needs it again. Move into the workspace folder using Step 3, then use the commands below. You do not need to reinstall the packages on every session.

On Windows, move into the workspace and use the environment’s full Python path again. To leave an activated macOS/Linux environment, type deactivate. That changes the terminal’s command selection; it does not delete your code or installed packages.

**macOS / Linux — new terminal, same folder**

```sh
# macOS / Linux — new terminal, same folder
source .venv/bin/activate
python first_experiment.py
```

**Expected:** The same saved program runs using the existing environment.

**Windows — new PowerShell, same folder**

```powershell
# Run windows — new powershell, same folder using the course Python environment
.\.venv\Scripts\python.exe first_experiment.py
```

**Expected:** The explicit environment interpreter runs the saved program without activation.

## When a step fails: match the symptom

“No such file” / “Cannot find path” for requirements-beginner.txt: check Step 3. The file must be in the current folder. Do not create an empty replacement file.

“No module named jax”: first check which Python is running. Repeat the interpreter-path check in Step 4, then install using that exact environment interpreter. A different terminal or editor may be selecting a different environment.

“No module named venv” or a message that ensurepip is unavailable: your Python distribution is missing its environment component. On Linux, use the distribution’s matching Python/venv package instructions; package names vary. This needs fixing before the environment can be created.

“No matching distribution found” or a failed jaxlib wheel download: record the Python version, operating system, CPU architecture, and full error. Check the JAX installation support table linked below. Keep the version pins and use only a Python release in the supported range; do not switch to TPU packages as a blind workaround. CPU platform support is not identical across operating systems; this repository’s Windows walkthrough is documented but not execution-validated.

A SyntaxError after pasting a terminal command: check whether you pasted it into a Python file or at the >>> prompt. Terminal commands and Python statements have different places to run.

Sum: $15.0$ followed by AssertionError: you changed the array length but left the old expected sum in the assertion. Update the expectation after deriving the new result; do not delete the check merely to hide the failure.

## Select the CPU backend before JAX loads

Set JAX_PLATFORMS before importing JAX so this starter uses the CPU runtime.

```python
import os
# Choose CPU before importing JAX for this first experiment.
os.environ["JAX_PLATFORMS"] = "cpu"
```

The environment variable must be set before the JAX import: JAX reads platform selection while initializing its runtime.

## Print the interpreter and JAX runtime details

Import Python platform information and JAX, then report the selected backend and devices.

```python
import platform
import jax
import numpy as np
print("Python:", platform.python_version())
print("JAX:", jax.__version__)
```

These lines identify the interpreter and backend that produced the result. They are environment evidence, not part of the sum calculation.

## Sum four JAX array elements and check the result

Print the NumPy version, create [0, 1, 2, 3] as float32, and compare its sum with the hand-computed value.

```python
print("NumPy:", np.__version__)
print("Backend:", jax.default_backend())
print("Devices:", jax.devices())
x = jax.numpy.arange(4, dtype=jax.numpy.float32)
print("Sum:", float(x.sum()))
assert float(x.sum()) == 6.0
```

The array sum is 0 + 1 + 2 + 3 = 6. The equality check is exact here because these small integers are represented exactly in float32. The assembled blocks include the same version and device diagnostics as the complete example.

## Run the example

```python
# Set up your learning workspace
import os
# Choose CPU before importing JAX for this first experiment.
os.environ["JAX_PLATFORMS"] = "cpu"
import platform
import jax
import numpy as np
print("Python:", platform.python_version())
print("JAX:", jax.__version__)
print("NumPy:", np.__version__)
print("Backend:", jax.default_backend())
print("Devices:", jax.devices())
x = jax.numpy.arange(4, dtype=jax.numpy.float32)
print("Sum:", float(x.sum()))
assert float(x.sum()) == 6.0
```

Expected: The package versions match the pins, Backend is cpu, and the last line is Sum: $6.0$. The exact Python patch and CPU device text depend on your installation.

## Workspace array verification across input elements

**Predict:** Before running, predict the element values and squared values of the 1D JAX array `x`.

![Workspace array verification across input elements](../../phases/00-welcome/01-set-up-your-learning-workspace/outputs/figure.svg)

**Recorded CPU computation**

### Read the figure

The horizontal axis is the element index; the vertical axis shows x=[0, 1, 2, 3] and x squared=[0, 1, 4, 9]. The figure is generated by the separate visual experiment and recorded as a reference; first_experiment.py prints the array sum and does not run Matplotlib.

### Connect it to the computation

The plot makes the four input values and their elementwise squares visible. It is generated by the lesson figure pipeline; the starter program itself computes and prints only the sum.

```python
# Plot the 1D verification array `x` and its elementwise square `x ** 2`:
sq_vals = x ** 2
visual_data = {'kind': 'bar', 'x': list(range(len(x))), 'labels': [f'x[{i}]' for i in range(len(x))], 'xlabel': 'element index', 'ylabel': 'array value', 'series': [{'label': 'x', 'y': [float(v) for v in x]}, {'label': 'x ** 2', 'y': [float(v) for v in sq_vals]}]}
```

## Recorded reference execution

CPU run: 2026-10-27T12:28:10.998141+00:00. JAX 0.9.2.

```text
Python: 3.14.3
JAX: 0.9.2
NumPy: 2.4.4
Backend: cpu
Devices: [CpuDevice(id=0)]
Sum: 6.0
Python: 3.14.3
JAX: 0.9.2
NumPy: 2.4.4
Backend: cpu
Devices: [CpuDevice(id=0)]
Sum: 6.0
Changed experiment sum: 15.0
PASS: welcome-01

```

## Make it yours

Edit the starter as shown in Step 6: change the array length from $4$ to $6$ and the expected sum from $6$ to $15$. Save, rerun, and keep my-first-run.txt plus a note explaining why the result changed.

### How to write this exercise — Step-by-step recipe & starter scaffold

**Inputs:**
- The saved experiment currently uses a JAX array and prints its total.
- The changed input contains six values.
- Save the observed result and explain why it changed.

**Outputs:**
- A modified runnable function and a first-run evidence note.

**Invariants to check:**
- The saved script is the source that was executed.
- The note reports an observed result.
- The command can be rerun from the experiment folder.

**Hints:**
- Make the requested edit in the starter file.
- Save terminal output before writing the explanation.

**Starter code scaffold (fill in the TODOs):**

```python
import jax.numpy as jnp

def changed_experiment():
    # TODO: create the changed input and report its total
    result = ...
    return result
```

<details><summary>Reference solution</summary>

```python
# Exercise solution
x = jax.numpy.arange(6, dtype=jax.numpy.float32)
print("Changed experiment sum:", float(x.sum()))
assert float(x.sum()) == 15.0
```

</details>

## Check your understanding

Which evidence lets another learner reproduce your environment?

1. A screenshot of the final number only
2. Python/package versions, device information, and the script
3. The name of your editor

<details><summary>Answer and explanation</summary>

**1. A screenshot of the final number only**

A screenshot records the result, but leaves the inputs and computation hidden. Give another learner the script and environment record so they can try to reproduce it.

**2. Python/package versions, device information, and the script (correct)**

The script records what ran; versions and device information record where it ran. CPU success is separate from TPU compatibility.

**3. The name of your editor**

The editor does not determine the numerical backend or package APIs. Record Python, package versions, devices and the runnable script.

</details>

## Diagnose the result

Use the troubleshooting section to identify the failing step. Keep the command and final error line, then verify the fix by repeating that step. Successful package installation, successful imports, and a correct numerical result are three separate checks.

## Carry forward

- Terminal commands run from a working folder; Python statements live in the .py file.
- The environment interpreter determines which installed packages the program uses.
- Expected values let you check the numerical result after a deliberate edit.
- Keep your code and results outside .venv so the environment can be recreated.

## Keep your evidence

Keep first_experiment.py, my-first-run.txt with package/device output, the environment interpreter path, and a note deriving the change from a sum of 6 to 15. Record a setup error and its repair if you encountered one.

Save your predictions, modified code, terminal outputs, and reasoning in your engineering portfolio.

## Primary references

- [Python downloads: choose Python 3.14 for your operating system](https://www.python.org/downloads/)
- [Python: creating and using virtual environments](https://docs.python.org/3.14/library/venv.html)
- [JAX: installation and supported platforms](https://docs.jax.dev/en/latest/installation.html)

