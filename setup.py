import os
import re
import subprocess

from setuptools import find_namespace_packages, setup


# Ensure we match the version set in optimum/intel/version.py
try:
    filepath = "optimum/intel/version.py"
    with open(filepath) as version_file:
        (__version__,) = re.findall('__version__ = "(.*)"', version_file.read())
    if __version__.endswith(".dev0"):
        dev_version_id = ""
        try:
            repo_root = os.path.dirname(os.path.realpath(__file__))
            dev_version_id = (
                subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=repo_root)  # nosec
                .strip()
                .decode()
            )
            dev_version_id = "+" + dev_version_id
        except subprocess.CalledProcessError:
            pass
        __version__ = __version__ + dev_version_id
except Exception as error:
    assert False, "Error: Could not open '%s' due %s\n" % (filepath, error)

INSTALL_REQUIRE = [
    "torch>=1.11",
    "transformers>=4.36.0,<4.42.0",
    "optimum~=1.20",
    "datasets>=1.4.0",
    "sentencepiece",
    "setuptools",
    "scipy",
    "onnx",
]

TESTS_REQUIRE = [
    "accelerate",
    "pytest>=7.2.0,<8.0.0",
    "parameterized",
    "Pillow",
    "evaluate",
    "diffusers",
    "py-cpuinfo",
    "sacremoses",
    "torchaudio",
    "rjieba",
    "timm",
    "invisible-watermark>=0.2.0",
    "transformers_stream_generator",
    "einops",
    "tiktoken",
    "sentence-transformers",
]

QUALITY_REQUIRE = ["black~=23.1", "ruff>=0.0.241"]

EXTRAS_REQUIRE = {
    "neural-compressor": ["neural-compressor>=2.2.0", "onnxruntime<1.15.0", "accelerate"],
    "openvino": ["openvino>=2023.3", "nncf>=2.11.0", "openvino-tokenizers[transformers]"],
    "nncf": ["nncf>=2.11.0"],
    "ipex": ["intel-extension-for-pytorch", "transformers>=4.39.0,<=4.41.2"],
    "diffusers": ["diffusers"],
    "quality": QUALITY_REQUIRE,
    "tests": TESTS_REQUIRE,
}

setup(
    name="optimum-intel",
    version=__version__,
    description="Optimum Library is an extension of the Hugging Face Transformers library, providing a framework to "
    "integrate third-party libraries from Hardware Partners and interface with their specific "
    "functionality.",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "License :: OSI Approved :: Apache Software License",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="transformers, quantization, pruning, knowledge distillation, optimization, training",
    url="https://www.intel.com",
    author="HuggingFace Inc. Special Ops Team",
    author_email="hardware@huggingface.co",
    license="Apache",
    packages=find_namespace_packages(include=["optimum*"]),
    install_requires=INSTALL_REQUIRE,
    extras_require=EXTRAS_REQUIRE,
    include_package_data=True,
    zip_safe=False,
    entry_points={"console_scripts": ["optimum-cli=optimum.commands.optimum_cli:main"]},
)
