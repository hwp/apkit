from setuptools import setup

setup(name='apkit',
      version='0.1',
      description='Python Audio Processing Kit',
      url='https://gitlab.idiap.ch/whe/apkit',
      author='Weipeng He',
      author_email='weipeng.he@idiap.ch',
      license='MIT',
      packages=['apkit'],
      install_requires=['scipy', 'numpy'],
      include_package_data=True,
      zip_safe=False)

