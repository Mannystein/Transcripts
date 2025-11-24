from setuptools import setup, Extension
import numpy as np

extension_mod = Extension("transcripts",
                         sources=["modules/helpers.c","modules/transcripts.c","modules/wrappers.c"],
                         include_dirs=[np.get_include()],
                         extra_compile_args=["-std=c99"])

setup(name="transcripts",
      version = "1.0.0",
      author = "Manuel Adams",
      author_email = "ma@uni-bonn.de",
      description="Transcript-based estimators for properties of interactions",
      long_description=open("README.md").read(),
      long_description_type="text/markdown",
      license ="MIT",
      classifiers=[
      "Programming Language :: Python :: 3",
      "Programming Language :: C",
      "License :: OSI Approved :: MIT License",
      "Operating System :: OS Independent",
          ],
      python_requires=">=3.8",
      include_dirs=[np.get_include()],
      ext_modules=[extension_mod],
)
