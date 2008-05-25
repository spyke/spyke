"""spyke installation script"""

from distutils.core import setup
import os

setup(name = 'spyke',
      version = '0.1',
      description = 'Multichannel spike viewer and sorter for Swindale Lab .srf files',
      author = 'Martin Spacek, Reza Lotun',
      author_email = 'mspacek at interchange ubc ca',
      url = 'http://swindale.ecc.ubc.ca',
      #long_description = '',
      packages = ['spyke', 'spyke.gui', 'spyke.gui.res'])
