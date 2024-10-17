from setuptools import setup, find_packages

setup(
    name="labelvid",
    version="0.1",
    packages=find_packages(),
    install_requires=["opencv-python"],
    entry_points={
        "console_scripts": [
            "labelvid=scripts.labelvid:main",  # Point to the main function
        ]
    },
)
