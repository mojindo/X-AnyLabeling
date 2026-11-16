import os
import subprocess
import sys
import tempfile
import threading
from io import StringIO
from typing import Tuple

from PyQt6.QtCore import QObject, pyqtSignal


from .utils import check_package_installed
from .validators import install_packages_with_timeout


class ExportEventRedirector(QObject):
    """Thread-safe export event redirector"""

    export_event_signal = pyqtSignal(str, dict)

    def __init__(self):
        super().__init__()

    def emit_export_event(self, event_type, data):
        """Safely emit export events from child thread to main thread"""
        self.export_event_signal.emit(event_type, data)


class ExportLogRedirector(QObject):
    """Thread-safe export log redirector"""

    log_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.log_stream = StringIO()

    def write(self, text):
        """Write text to log stream and emit signal if not empty"""
        if text.strip():
            self.log_signal.emit(text)

    def flush(self):
        """Flush the log stream"""
        pass


def validate_onnx_export_environment():
    # Modern Ultralytics works with current onnx; keep onnxslim preferred but
    # not a hard block when install fails (export can use simplify=False).
    required_packages = ["onnx", "onnxruntime"]
    missing_packages = []
    package_mapping = {
        "onnx": "onnx>=1.12.0",
        "onnxruntime": "onnxruntime",
        "onnxslim": "onnxslim>=0.1.59",
    }

    for package in required_packages:
        if not check_package_installed(package):
            missing_packages.append(package_mapping[package])

    # Prefer onnxslim when missing, but do not abort if later install fails
    if not check_package_installed("onnxslim"):
        missing_packages.append(package_mapping["onnxslim"])

    return missing_packages


def validate_openvino_export_environment():
    required_packages = ["openvino"]
    missing_packages = []
    package_mapping = {"openvino": "openvino>=2024.0.0"}

    for package in required_packages:
        if not check_package_installed(package):
            missing_packages.append(package_mapping[package])

    return missing_packages


def validate_tensorrt_export_environment():
    required_packages = ["tensorrt"]
    missing_packages = []
    package_mapping = {"tensorrt": "tensorrt>7.0.0,!=10.1.0"}

    for package in required_packages:
        if not check_package_installed(package):
            missing_packages.append(package_mapping[package])

    return missing_packages


def validate_coreml_export_environment():
    required_packages = ["coremltools"]
    missing_packages = []
    package_mapping = {"coremltools": "coremltools>=8.0"}

    for package in required_packages:
        if not check_package_installed(package):
            missing_packages.append(package_mapping[package])

    return missing_packages


def validate_tensorflow_export_environment():
    required_packages = ["tensorflow"]
    missing_packages = []
    package_mapping = {"tensorflow": "tensorflow>=2.0.0"}

    for package in required_packages:
        if not check_package_installed(package):
            missing_packages.append(package_mapping[package])

    return missing_packages


def validate_paddle_export_environment():
    required_packages = ["paddlepaddle", "x2paddle"]
    missing_packages = []
    package_mapping = {
        "paddlepaddle": "paddlepaddle-gpu",
        "x2paddle": "x2paddle",
    }

    for package in required_packages:
        if not check_package_installed(package):
            missing_packages.append(package_mapping[package])

    return missing_packages


def validate_mnn_export_environment():
    required_packages = ["MNN"]
    missing_packages = []
    package_mapping = {"MNN": "MNN>=2.9.6"}

    for package in required_packages:
        if not check_package_installed(package):
            missing_packages.append(package_mapping[package])

    return missing_packages


def validate_ncnn_export_environment():
    required_packages = ["ncnn"]
    missing_packages = []
    package_mapping = {"ncnn": "ncnn"}

    for package in required_packages:
        if not check_package_installed(package):
            missing_packages.append(package_mapping[package])

    return missing_packages


def validate_imx500_export_environment():
    required_packages = ["imx500-converter", "mct-quantizers"]
    missing_packages = []
    package_mapping = {
        "imx500-converter": "imx500-converter[pt]>=3.16.1",
        "mct-quantizers": "mct-quantizers>=1.6.0",
    }

    for package in required_packages:
        if not check_package_installed(package):
            missing_packages.append(package_mapping[package])

    return missing_packages


def validate_rknn_export_environment():
    required_packages = ["rknn-toolkit2"]
    missing_packages = []
    package_mapping = {"rknn-toolkit2": "rknn-toolkit2"}

    for package in required_packages:
        if not check_package_installed(package):
            missing_packages.append(package_mapping[package])

    return missing_packages


def get_export_validator(export_format):
    validators = {
        "onnx": validate_onnx_export_environment,
        "openvino": validate_openvino_export_environment,
        "engine": validate_tensorrt_export_environment,
        "coreml": validate_coreml_export_environment,
        "saved_model": validate_tensorflow_export_environment,
        "pb": validate_tensorflow_export_environment,
        "tflite": validate_tensorflow_export_environment,
        "edgetpu": validate_tensorflow_export_environment,
        "tfjs": validate_tensorflow_export_environment,
        "paddle": validate_paddle_export_environment,
        "mnn": validate_mnn_export_environment,
        "ncnn": validate_ncnn_export_environment,
        "imx": validate_imx500_export_environment,
        "rknn": validate_rknn_export_environment,
        "torchscript": lambda: [],
    }
    return validators.get(export_format, lambda: [])


class ExportManager:
    def __init__(self):
        self.is_exporting = False
        self.callbacks = []
        self.export_thread = None

    def notify_callbacks(self, event_type: str, data: dict):
        for callback in self.callbacks:
            try:
                callback(event_type, data)
            except Exception as e:
                print(f"Error in export callback: {e}")

    def start_export(
        self,
        project_path: str,
        export_format: str = "onnx",
        *,
        weights_file: str = "best.pt",
        output_name: str = "",
        imgsz: int = 640,
        python_executable: str = "",
    ) -> Tuple[bool, str]:
        if self.is_exporting:
            return False, "Export already in progress"

        weights_file = os.path.basename(weights_file or "best.pt")
        weights_path = os.path.join(project_path, "weights", weights_file)
        if not os.path.exists(weights_path):
            return False, f"Model weights not found at: {weights_path}"

        self.is_exporting = True
        self.export_thread = threading.Thread(
            target=self._export_worker,
            args=(
                weights_path,
                export_format,
                output_name,
                imgsz,
                python_executable or "",
            ),
        )
        self.export_thread.start()
        return True, "Export started successfully"

    def _export_worker(
        self,
        weights_path: str,
        export_format: str,
        output_name: str = "",
        imgsz: int = 640,
        python_executable: str = "",
    ):
        py_exe = (python_executable or "").strip()
        if py_exe:
            self._export_worker_subprocess(
                weights_path, export_format, output_name, imgsz, py_exe
            )
            return
        try:
            self.notify_callbacks(
                "export_started",
                {"weights_path": weights_path, "format": export_format},
            )
            self.notify_callbacks(
                "export_log", {"message": "Checking export environment..."}
            )
            missing_packages = get_export_validator(export_format)()
            use_simplify = True
            if missing_packages:
                self.notify_callbacks(
                    "export_log",
                    {
                        "message": f"Missing required packages: {', '.join(missing_packages)}"
                    },
                )
                self.notify_callbacks(
                    "export_log",
                    {"message": "Attempting to install missing packages..."},
                )
                success, stdout, stderr = install_packages_with_timeout(
                    missing_packages, timeout=300
                )
                if not success:
                    if not check_package_installed("onnx"):
                        error_msg = (
                            f"Failed to install required packages: "
                            f"{', '.join(missing_packages)}. "
                            "Please manually install: pip install "
                            '"onnx>=1.12.0" onnxruntime\n'
                            f"Detail: {(stderr or stdout or '').strip()}"
                        )
                        self.notify_callbacks(
                            "export_error", {"error": error_msg}
                        )
                        return
                    use_simplify = False
                    self.notify_callbacks(
                        "export_log",
                        {
                            "message": (
                                "Optional packages missing; exporting ONNX "
                                "with simplify=False"
                            )
                        },
                    )
                else:
                    self.notify_callbacks(
                        "export_log",
                        {
                            "message": "Required packages installed successfully"
                        },
                    )
                    use_simplify = check_package_installed("onnxslim")
            else:
                self.notify_callbacks(
                    "export_log",
                    {"message": "All required packages are available"},
                )
                use_simplify = check_package_installed("onnxslim")

            original_stdout = sys.stdout
            original_stderr = sys.stderr

            log_redirector = ExportLogRedirector()
            sys.stdout = log_redirector
            sys.stderr = log_redirector

            try:
                from ultralytics import YOLO

                self.notify_callbacks(
                    "export_log",
                    {"message": f"Loading model from {weights_path}"},
                )
                model = YOLO(weights_path)

                self.notify_callbacks(
                    "export_log",
                    {
                        "message": f"Starting export to {export_format} format..."
                    },
                )
                export_kwargs = {"format": export_format}
                if export_format == "onnx":
                    if imgsz:
                        export_kwargs["imgsz"] = int(imgsz)
                    export_kwargs["simplify"] = bool(use_simplify)
                results = model.export(**export_kwargs)

                exported_path = (
                    results if isinstance(results, str) else str(results)
                )
                if not exported_path:
                    weights_dir = os.path.dirname(weights_path)
                    model_name = os.path.splitext(
                        os.path.basename(weights_path)
                    )[0]
                    exported_path = os.path.join(
                        weights_dir, f"{model_name}.{export_format}"
                    )

                if not os.path.exists(exported_path):
                    possible_path = weights_path.replace(
                        ".pt", f".{export_format}"
                    )
                    if os.path.exists(possible_path):
                        exported_path = possible_path

                if (
                    export_format == "onnx"
                    and output_name
                    and os.path.exists(exported_path)
                ):
                    weights_dir = os.path.dirname(weights_path)
                    target = os.path.join(
                        weights_dir, os.path.basename(output_name)
                    )
                    if os.path.abspath(exported_path) != os.path.abspath(
                        target
                    ):
                        if os.path.exists(target):
                            os.remove(target)
                        os.replace(exported_path, target)
                    exported_path = target
                    self.notify_callbacks(
                        "export_log",
                        {"message": f"Renamed ONNX to: {exported_path}"},
                    )

                if os.path.exists(exported_path):
                    self.notify_callbacks(
                        "export_completed",
                        {
                            "exported_path": exported_path,
                            "format": export_format,
                        },
                    )
                else:
                    self.notify_callbacks(
                        "export_error",
                        {
                            "error": "Export completed but output file not found"
                        },
                    )
            finally:
                sys.stdout = original_stdout
                sys.stderr = original_stderr
        except Exception as e:
            self.notify_callbacks(
                "export_error",
                {"error": f"Unexpected error during export: {str(e)}"},
            )
        finally:
            self.is_exporting = False

    def _export_worker_subprocess(
        self,
        weights_path: str,
        export_format: str,
        output_name: str,
        imgsz: int,
        python_executable: str,
    ):
        script_path = ""
        try:
            self.notify_callbacks(
                "export_started",
                {
                    "weights_path": weights_path,
                    "format": export_format,
                    "python": python_executable,
                },
            )
            if not os.path.isfile(python_executable):
                self.notify_callbacks(
                    "export_error",
                    {
                        "error": f"Training Python not found: {python_executable}"
                    },
                )
                return

            self.notify_callbacks(
                "export_log",
                {
                    "message": f"Export via interpreter: {python_executable}"
                },
            )
            out_name = (output_name or "").strip()
            script = f"""# -*- coding: utf-8 -*-
import io
import os
import sys
if sys.platform.startswith("win"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
from ultralytics import YOLO
weights = {weights_path!r}
fmt = {export_format!r}
imgsz = {int(imgsz)}
output_name = {out_name!r}
model = YOLO(weights)
kwargs = {{"format": fmt}}
if fmt == "onnx":
    kwargs["imgsz"] = imgsz
    kwargs["simplify"] = True
result = model.export(**kwargs)
exported = result if isinstance(result, str) else str(result)
if not exported or not os.path.exists(exported):
    alt = weights.replace(".pt", f".{{fmt}}")
    if os.path.exists(alt):
        exported = alt
if fmt == "onnx" and output_name and exported and os.path.exists(exported):
    target = os.path.join(os.path.dirname(weights), os.path.basename(output_name))
    if os.path.abspath(exported) != os.path.abspath(target):
        if os.path.exists(target):
            os.remove(target)
        os.replace(exported, target)
    exported = target
print("EXPORT_OK:" + exported, flush=True)
"""
            fd, script_path = tempfile.mkstemp(
                suffix="_export.py", prefix="ual_"
            )
            os.close(fd)
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(script)

            proc = subprocess.Popen(
                [python_executable, script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            exported_path = ""
            assert proc.stdout is not None
            for line in proc.stdout:
                cleaned = (line or "").rstrip()
                if cleaned:
                    self.notify_callbacks(
                        "export_log", {"message": cleaned}
                    )
                if cleaned.startswith("EXPORT_OK:"):
                    exported_path = cleaned[len("EXPORT_OK:") :].strip()
            code = proc.wait()
            if code == 0 and exported_path and os.path.exists(exported_path):
                self.notify_callbacks(
                    "export_completed",
                    {
                        "exported_path": exported_path,
                        "format": export_format,
                    },
                )
            elif code == 0:
                self.notify_callbacks(
                    "export_error",
                    {
                        "error": "Export completed but output file not found"
                    },
                )
            else:
                self.notify_callbacks(
                    "export_error",
                    {
                        "error": f"Export process exited with code {code}"
                    },
                )
        except Exception as e:
            self.notify_callbacks(
                "export_error",
                {"error": f"Unexpected error during export: {str(e)}"},
            )
        finally:
            self.is_exporting = False
            if script_path:
                try:
                    os.remove(script_path)
                except OSError:
                    pass

    def stop_export(self) -> bool:
        if not self.is_exporting:
            return False

        self.is_exporting = False
        if self.export_thread and self.export_thread.is_alive():
            self.export_thread.join(timeout=5)

        self.notify_callbacks("export_stopped", {})
        return True


_export_manager = None


def get_export_manager() -> ExportManager:
    global _export_manager
    if _export_manager is None:
        _export_manager = ExportManager()
    return _export_manager


def export_model(
    project_path: str, export_format: str = "onnx"
) -> Tuple[bool, str]:
    manager = get_export_manager()
    return manager.start_export(project_path, export_format)
