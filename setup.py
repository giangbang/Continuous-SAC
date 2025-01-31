from setuptools import setup, find_packages
setup(
    name = "sac_continuous",
    version = "0.0.1",
    description = ("Reproduce results from Continuous SAC paper."),
    packages=find_packages(),
    install_requires=[
        "gymnasium",
        "numpy",
        "matplotlib",
        "torch"
    ],
    entry_points={
        'console_scripts': [
            'sac_continuous = train:src.main',
        ],
    },
)