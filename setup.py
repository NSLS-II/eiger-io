from setuptools import setup
import versioneer

setup(version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      name='eiger_io',
      packages=['eiger_io'])

