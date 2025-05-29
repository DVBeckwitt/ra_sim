from setuptools import setup, find_packages
import os

# Read the long description from README.md if it exists
long_description = ""
if os.path.isfile("README.md"):
    with open("README.md", "r", encoding="utf-8") as f:
        long_description = f.read()

setup(
    name='ra_sim',
    version='0.1.0',
    packages=find_packages(),
    author='Your Name',
    author_email='your.email@example.com',
    description='A Python package for simulating and analyzing XRD patterns',
    long_description=long_description,
    long_description_content_type='text/markdown',
    license='GPLv3',
    keywords=['XRD', 'diffraction', 'image processing', 'crystallography'],
    url='https://github.com/YourUsername/myxrdanalysis',  # Update to your project URL

    # These are the runtime dependencies for your package:
    install_requires=[
        'numpy>=1.18.0',
        'matplotlib>=3.0.0',
        'pyFAI>=0.19.0',
        'fabio>=0.11.0',
        'scipy>=1.4.0',
        'sympy>=1.5.1',
        'pandas>=1.0.0',
        'numba>=0.50.0',
        'spglib>=1.9.0',
        'bayesian-optimization>=1.2.0',
        'tifffile>=2020.9.3',
        'docopt>=0.6.2',
        'logbook>=1.5.3',
        'scikit-image>=0.17.0',  # For skimage.feature, skimage.metrics, etc.
        'Pillow>=7.0.0',         # For PIL-based image I/O
        'ipython>=7.0.0',        # For IPython.display
        'jupyter>=1.0.0',        # Needed if using pyFAI.gui.jupyter
    ],

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Topic :: Scientific/Engineering :: Image Processing',
    ],

    python_requires='>=3.8',
    include_package_data=True,
    zip_safe=False,
)
