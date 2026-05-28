from setuptools import setup, find_packages

setup(
    name="entexbert2",
    version="0.1.0",
    description="Fine-tuning a Transformer model on EN-TEx data built off a modified DNABERT-2 backbone",
    author="Amy Metrick",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    include_package_data=True,
)