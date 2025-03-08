from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="arborium",
    version="0.1.3",
    author="Rishabh Mandayam",
    author_email="rishabh.mandayam@gmail.com",
    description="A tree visualization and analysis package for XGBoost models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/arborium",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
    ],
    python_requires=">=3.6",
    install_requires=[
        "numpy",
        "scikit-learn",
        "ipython",
    ],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "flake8",
        ],
        "xgboost": ["xgboost>=1.0.0"],
    },
    include_package_data=True,
) 