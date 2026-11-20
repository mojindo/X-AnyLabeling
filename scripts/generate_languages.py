import os
import glob
import sys
import shutil
import subprocess
def find_lrelease() -> str:
    """Return an available Qt Linguist release compiler."""
    candidates = ("lrelease", "lrelease-qt6", "pyside6-lrelease")
    for candidate in candidates:
        executable = shutil.which(candidate)
        if executable:
            return executable
    raise RuntimeError(
        "No Qt translation compiler found. Install lrelease or ensure "
        "pyside6-lrelease is available in the active environment."
    )


def compile_resources(output: str, qrc: str) -> None:
    """Compile a .qrc file to a PyQt6-compatible resources.py."""

    def normalize_imports(path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        content = content.replace("from PySide6", "from PyQt6")
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

    def add_rcc_commands(commands, base_command, needs_rewrite):
        # Force zlib resources when supported. Newer RCC builds may emit zstd
        # entries by default, which are not readable in some Windows Qt runtimes.
        commands.append(
            (
                [*base_command, "--compress-algo", "zlib", "-o", output, qrc],
                needs_rewrite,
            )
        )
        commands.append(([*base_command, "-o", output, qrc], needs_rewrite))

    commands = []
    add_rcc_commands(
        commands, [sys.executable, "-m", "PyQt6.pyrcc_main"], False
    )
    add_rcc_commands(commands, ["pyrcc6"], False)
    add_rcc_commands(commands, ["pyside6-rcc"], True)
    add_rcc_commands(commands, ["rcc", "-g", "python"], True)
    lrelease = shutil.which("lrelease")
    if lrelease:
        sibling_rcc = os.path.join(os.path.dirname(lrelease), "rcc")
        add_rcc_commands(commands, [sibling_rcc, "-g", "python"], True)
    for command, needs_rewrite in commands:
        executable = command[0]
        if executable != sys.executable and not shutil.which(executable):
            continue
        result = subprocess.run(command, stderr=subprocess.DEVNULL)
        if result.returncode != 0:
            continue
        if needs_rewrite:
            normalize_imports(output)
        return
    print(
        "Error: no Qt resource compiler found. Tried python -m PyQt6.pyrcc_main, pyrcc6, pyside6-rcc, rcc -g python, and lrelease-sibling rcc."
    )


supported_languages = ["en_US", "zh_CN", "ja_JP", "ko_KR"]
translations_path = "anylabeling/resources/translations"
lrelease = find_lrelease()

def run_pylupdate(ts_path: str) -> None:
    """Extract translatable strings without blowing the Windows command-line limit."""
    pylupdate = shutil.which("pylupdate6") or "pylupdate6"
    sources = []
    if os.path.isdir("anylabeling"):
        sources.append("anylabeling")
    # Root-level entry scripts (if any) with tr()
    for py in glob.glob("*.py"):
        sources.append(py)
    if not sources:
        raise RuntimeError("No Python sources found for pylupdate6")
    result = subprocess.run(
        [pylupdate, "--no-obsolete", *sources, "-ts", ts_path],
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"pylupdate6 failed for {ts_path}")


for language in supported_languages:
    # Translate all .ui files into .py files
    ui_files = glob.glob(os.path.join("anylabeling", "**", "*.ui"), recursive=True)
    for ui_file in ui_files:
        py_file = os.path.splitext(ui_file)[0] + "_ui.py"
        command = f"pyuic6 -x {ui_file} -o {py_file}"
        os.system(command)

    run_pylupdate(f"{translations_path}/{language}.ts")

    # Compile the .ts file into a .qm file
    translation_file = f"{translations_path}/{language}.ts"
    subprocess.run([lrelease, translation_file], check=True)

compile_resources(
    output="anylabeling/resources/resources.py",
    qrc="anylabeling/resources/resources.qrc",
)
