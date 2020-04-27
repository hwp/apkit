from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='apkit',
    version='0.3.1',
    author='Weipeng He',
    author_email='heweipeng@gmail.com',
    description='Python Audio Processing Kit',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/hwp/apkit',
    license='MIT',
    packages=['apkit'],
    install_requires=['scipy', 'numpy'],
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)

