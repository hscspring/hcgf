import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="hcgf",
    version="0.0.1",
    author="Yam",
    author_email="haoshaochun@gmail.com",
    description="Humanable ChatGPT/GLM Fine-tuning.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hscspring/hcgf",
    include_package_data=True,
    # default is `setup.py` path, so do not need a `package_dir` attr
    # if another dir, should be declared by `package_dir`
    packages=setuptools.find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    install_requires=[
    ],
    package_data={
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)