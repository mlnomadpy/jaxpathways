# Set up your learning workspace

Phase 00: Setup & first steps · about 40 minutes · CPU

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

*Conceptual / analytic teaching diagram; not a recorded benchmark.*

Read downward: a command runs from a working folder, selects an interpreter, opens the file and imports its packages before computing. Arrows are dependencies, not measured time. A file-path failure happens before JAX arithmetic; an import failure points to the selected environment.

### Pause and reason

The notebook imports JAX, but the terminal script does not. What should you compare first?

<details><summary>Compare your reasoning</summary>

Compare the Python executables used by the notebook kernel and terminal. They can belong to different environments. Install into, or select, the intended environment only after identifying that difference.

</details>

## 1. Download and unpack your workspace

Use the “Download the beginner workspace” button above. Open your Downloads folder and unzip jax-start-here.zip. On Windows, right-click it and choose Extract All; on macOS, double-click it. Open the extracted jax-start-here folder. Do not work inside the ZIP preview.

You should see first_experiment.py, requirements-cpu.txt, and START-HERE.md. first_experiment.py is the program you will run. requirements-cpu.txt lists the package versions to install. START-HERE.md is a copy of these instructions. Keep your own files alongside them, outside .venv. You do not need to clone the repository for this lesson.

## 2. Check that Python is installed

Open a terminal. On macOS, press Command–Space, type Terminal, and press Return. On Windows, open Start, type PowerShell, and open it. On Linux, open your desktop’s Terminal application.

Choose the command for your operating system below. Type or paste one line, then press Enter. Do not type a prompt symbol such as $ or PS>. If you see >>>, you are already inside the Python interpreter; type `exit()` and press Enter to return to the terminal before using these commands.

The course CPU examples were checked with Python 3.14.3. These commands select Python $3.14$. If you get “command not found” or “not recognized,” Python $3.14$ is not available through that command yet. Open the official Python downloads link in the references, select a Python $3.14$ installer for macOS or Windows, and follow the installer. On Windows, the launcher/install manager supplies py; on macOS, the Python installer supplies python3.14. Close and reopen the terminal after installing, then retry the version check. Linux installation depends on the distribution: use its Python $3.14$ installation instructions and return here once the version check works. Do not continue by substituting an old system Python without checking compatibility.

**macOS / Linux — terminal**

```sh
python3.14 --version
```

**Expected:** A line beginning Python 3.14. The last patch number may differ from the tested 3.14.3.

**Windows — PowerShell**

```powershell
py -3.14 --version
```

**Expected:** A line beginning Python 3.14. If py exists but cannot find 3.14, install that Python version first.

## 3. Put the terminal in the right folder

The program and requirements file are in the extracted workspace. A command using a filename looks for it relative to the terminal’s current folder. We need the terminal to be in jax-start-here.

macOS: type cd followed by a space, drag the extracted jax-start-here folder from Finder into the terminal, then press Enter. Dragging inserts its actual path, including spaces correctly. Linux: use cd with the actual folder path in quotes; your file manager may offer “Open in Terminal.” Windows: open the extracted folder in File Explorer, click its address bar, type powershell, and press Enter. That opens PowerShell in this folder.

Run the appropriate check below. If requirements-cpu.txt is missing, move into the extracted folder containing that file before continuing. “Repository root” in later lessons means the top folder of the full course checkout; this beginner workspace is enough for the current experiment.

**macOS / Linux — check folder and files**

```sh
pwd
ls
```

**Expected:** The path ends with your extracted workspace folder, and the file list includes first_experiment.py and requirements-cpu.txt.

**Windows — check folder and files**

```powershell
Get-Location
Get-ChildItem
```

**Expected:** The displayed folder contains first_experiment.py and requirements-cpu.txt.

## 4. Create the environment and install the packages

Run the lines for your operating system in order, staying in the workspace folder. The venv command creates .venv using the Python version you just selected. It often finishes without printing anything. That is normal; wait until the terminal is ready for another command. If you already have a .venv from another Python version in this folder, use a fresh extracted workspace for this walkthrough.

On macOS/Linux, source loads the activation script into this terminal. You may see (.venv) at the start of the prompt. The important check is sys.executable: it should point inside this workspace’s .venv/bin. On Windows we use .venv\Scripts\python.exe explicitly, so no activation script or execution-policy change is needed.

The pip command is the package installer. -$r$ means “read the package list from this file.” It downloads and installs packages into the selected environment. Let it finish. Download messages are normal; a traceback or an ERROR means the step did not succeed. The final import check below is more useful than guessing from the download log. The pins describe the tested CPU course setup; accelerator installation is a later lesson.

**macOS / Linux — run one line at a time**

```sh
python3.14 -m venv .venv
source .venv/bin/activate
python -c "import sys; print(sys.executable)"
python -m pip install -r requirements-cpu.txt
python -c "import jax, numpy, optax; print(jax.__version__, numpy.__version__, optax.__version__)"
```

**Expected:** The interpreter path includes .venv/bin/python. The final line is 0.9.2 2.4.4 0.2.8.

**Windows — run one line at a time**

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -c "import sys; print(sys.executable)"
.\.venv\Scripts\python.exe -m pip install -r requirements-cpu.txt
.\.venv\Scripts\python.exe -c "import jax, numpy, optax; print(jax.__version__, numpy.__version__, optax.__version__)"
```

**Expected:** The interpreter path includes .venv\Scripts\python.exe. The final line is 0.9.2 2.4.4 0.2.8. These PowerShell instructions have not been executed on Windows in this project.

## 5. Run your first saved program

Open first_experiment.py in a text/code editor if you want to see its contents. The same Python program is shown below in “Run the example.” Python code belongs in that file; the commands below belong in the terminal. The downloaded file is already saved, so you can run it without copying any code.

Run the command for your operating system. The program prints the Python and package versions, the backend, the devices, and a sum. The starter explicitly selects CPU before importing JAX, so the expected backend here is cpu. The device description can vary; the final result should be Sum: $6.0$. The program constructs the numbers $0$, $1$, $2$, $3$; adding them gives $6$. This known result checks that we ran a numerical computation, not only installed a package.

If you see a traceback, read the final line and compare it with the troubleshooting section. Do not count a failed command as a completed experiment.

**macOS / Linux — activated terminal**

```sh
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
.\.venv\Scripts\python.exe first_experiment.py
```

**Expected:** The same version/backend/result lines. CPU device wording may differ.

## 6. Make one change and save your result

In the editor, find `arange(4, ...)` in first_experiment.py and change $4$ to $6$. The array now contains $0$ through $5$. Predict their sum before running. Also change the final assert from `== 6.0` to `== 15.0`, because that assertion is the program’s expected-result check. Save the file with its .py extension; a filename such as first_experiment.py.txt will not match the run command.

Run it again using the Step $5$ command. The final line should be Sum: $15.0$. If it still says $6.0$, check that you saved the file and that the terminal is in the same workspace folder you edited.

After that successful run, the commands below run it once more and save its printed output to my-first-run.txt. Open that text file and check that the versions, CPU information, and sum are present. Add a note in your editor explaining the one change and your prediction. The > symbol saves output to a file and replaces its previous contents; choose another filename when you want to keep multiple runs.

**macOS / Linux — save output**

```sh
python first_experiment.py > my-first-run.txt
```

**Expected:** my-first-run.txt appears in the workspace and ends with Sum: 15.0 after your saved edit.

**Windows — save output**

```powershell
.\.venv\Scripts\python.exe first_experiment.py > my-first-run.txt
```

**Expected:** my-first-run.txt contains the program output after your saved edit.

## 7. Resume when you open a new terminal

The .venv folder stays on disk when you close the terminal. On macOS/Linux, activation only applied to the terminal you used; a new terminal needs it again. Move into the workspace folder using Step $3$, then use the commands below. You do not need to reinstall the packages on every session.

On Windows, move into the workspace and use the environment’s full Python path again. To leave an activated macOS/Linux environment, type deactivate. That changes the terminal’s command selection; it does not delete your code or installed packages.

**macOS / Linux — new terminal, same folder**

```sh
source .venv/bin/activate
python first_experiment.py
```

**Expected:** The same saved program runs using the existing environment.

**Windows — new PowerShell, same folder**

```powershell
.\.venv\Scripts\python.exe first_experiment.py
```

**Expected:** The explicit environment interpreter runs the saved program without activation.

## When a step fails: match the symptom

“No such file” / “Cannot find path” for requirements-cpu.txt: check Step $3$. The file must be in the current folder. Do not create an empty replacement file.

“No module named jax”: first check which Python is running. Repeat the interpreter-path check in Step $4$, then install using that exact environment interpreter. A different terminal or editor may be selecting a different environment.

“No module named venv” or a message that ensurepip is unavailable: your Python distribution is missing its environment component. On Linux, use the distribution’s matching Python/venv package instructions; package names vary. This needs fixing before the environment can be created.

“No matching distribution found” or a failed jaxlib wheel download: record the Python version, operating system, CPU architecture, and full error. Check the JAX installation support table linked below. Do not remove the version pins or switch to TPU packages as a blind workaround. CPU platform support is not identical across operating systems; this repository’s Windows walkthrough is documented but not execution-validated.

A SyntaxError after pasting a terminal command: check whether you pasted it into a Python file or at the >>> prompt. Terminal commands and Python statements have different places to run.

Sum: $15.0$ followed by AssertionError: you changed the array length but left the old expected sum in the assertion. Update the expectation after deriving the new result; do not delete the check merely to hide the failure.

## Run the example

```python
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

## From a saved file to a verified result

**Predict:** Where does the code run, and which output confirms the calculation?

![From a saved file to a verified result](../../phases/00-welcome/01-set-up-your-learning-workspace/outputs/figure.svg)

**Conceptual diagram**

### Read the figure

Read the boxes from top to bottom. They follow one experiment from choosing a Python environment to reading a computed result. The arrows mean “do this next”; their lengths do not represent time.

The saved file is the connection between your editor and Python. Saving text alone does not run it. Running the file asks JAX to create the array and compute its sum on the CPU.

### Connect it to the computation

The last box is where you check whether the setup worked: the example adds $0+1+2+3=6$, then checks that result with an assertion. A successful installation message only gets you to the beginning of this chain; the computed sum gets you to the end.

If your result is missing, walk backward through the boxes: did Python run the saved file, did it use the activated environment, and did that environment import JAX? This is a workflow diagram. The recorded output below supplies the separate evidence that the example executed.

## Recorded reference execution

CPU run: 2026-10-06T15:39:03.029200+00:00. JAX 0.9.2.

```text
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

Edit the starter as shown in Step $6$: change the array length from $4$ to $6$ and the expected sum from $6$ to $15$. Save, rerun, and keep my-first-run.txt plus a note explaining why the result changed.

<details><summary>Reference solution</summary>

```python
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

Python/package versions, device information, and the script

The script records what ran; versions and device information record where it ran. CPU success does not establish TPU compatibility.

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

Keep predictions, modified code, observed results, and reasoning. A checkpoint alone does not demonstrate the exercise.

## Primary references

- [Python downloads: choose Python 3.14 for your operating system](https://www.python.org/downloads/)
- [Python: creating and using virtual environments](https://docs.python.org/3.14/library/venv.html)
- [JAX: installation and supported platforms](https://docs.jax.dev/en/latest/installation.html)

