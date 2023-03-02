import re

from setuptools import find_namespace_packages, setup


# Ensure we match the version set in optimum/intel/version.py
try:
    filepath = "optimum/intel/version.py"
    with open(filepath) as version_file:
        (__version__,) = re.findall('__version__ = "(.*)"', version_file.read())
except Exception as error:
    assert False, "Error: Could not open '%s' due %s\n" % (filepath, error)

INSTALL_REQUIRE = [
    "optimum>=1.7.0",
    "transformers>=4.20.0",
    "datasets>=1.4.0",
    "torch",
    "sentencepiece",
    "scipy",
]

TESTS_REQUIRE = ["pytest", "parameterized", "Pillow", "evaluate", "diffusers"]

QUALITY_REQUIRE = ["black==22.3", "isort>=5.5.4"]

EXTRAS_REQUIRE = {
    "neural-compressor": ["neural-compressor>=2.0.0", "onnx", "onnxruntime"],
    "openvino": ["openvino>=2023.0.0.dev20230217", "onnx", "onnxruntime"],
    "nncf": ["nncf>=2.4.0", "openvino-dev>=2023.0.0.dev20230217"],
    "ipex": ["intel_extension_for_pytorch"],
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
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
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
)
