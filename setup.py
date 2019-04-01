from setuptools import setup

setup(name='apkit',
      version='0.2',
      description='Python Audio Processing Kit',
      url='https://github.com/hwp/apkit',
      author='Weipeng He',
      author_email='heweipeng@gmail.com',
      license='MIT',
      packages=['apkit'],
      install_requires=['scipy', 'numpy'],
      include_package_data=True,
      zip_safe=False)

