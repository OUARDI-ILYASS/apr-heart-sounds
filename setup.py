"""Editable install so that `import src.features.mfcc` works from anywhere.

    pip install -e .

Not strictly required - scripts/_bootstrap.py injects the project root into
sys.path - but it makes the package importable from notebooks, the REPL and
pytest without path juggling.
"""

from setuptools import setup, find_packages

setup(
    name="apr-heart-sounds",
    version="1.0.0",
    description="Interpretable phonocardiogram classification "
                "(Audio Pattern Recognition course project)",
    packages=find_packages(include=["src", "src.*"]),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24", "scipy>=1.10", "pandas>=2.0",
        "scikit-learn>=1.3", "joblib>=1.3",
        "librosa>=0.10", "soundfile>=0.12", "PyWavelets>=1.4",
        "torch>=2.0", "shap>=0.44",
        "matplotlib>=3.7", "PyYAML>=6.0", "tqdm>=4.65",
    ],
    extras_require={"dev": ["pytest>=7.4"]},
)
