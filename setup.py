"""Python setup.py for project_name package"""

import io
import os
from setuptools import find_packages, setup

version_dict = {}
with open("src/version.py") as f:
    code = compile(f.read(), f.name, "exec")
    exec(code, version_dict)


def read(*paths, **kwargs):
    """Read the contents of a text file safely.
    >>> read("VlmFinetune", "VERSION")
    '0.1.0'
    >>> read("README.md")
    ...
    """

    content = ""
    with io.open(
        os.path.join(os.path.dirname(__file__), *paths),
        encoding=kwargs.get("encoding", "utf8"),
    ) as open_file:
        content = open_file.read().strip()
    return content


def read_requirements(path):
    return [
        line.strip()
        for line in read(path).split("\n")
        if not line.startswith(('"', "#", "-", "git+"))
    ]


setup(
    name="VlmFinetune",
    version=version_dict["version"],
    description="A tool for fine-tuning Vision Language Model",
    url="https://github.com/sandy1990418/Finetune-Qwen2.5-VL.git",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    author="Sandy Chen",
    python_requires=">=3.9",
    packages=find_packages(exclude=["tests", ".github"]),
    install_requires=read_requirements("requirements.txt"),
    # entry_points={"console_scripts": ["project_name = project_name.__main__:main"]},
    license="Apache License 2.0",
    classifiers=[  # 建議加入分類資訊
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    keywords="vlm, finetune, qwen, vision-language-model",
    # extras_require={"test": read_requirements("requirements-test.txt")},
)
