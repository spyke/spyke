"""spyke installation script"""

from distutils.core import setup
import os

setup(name = 'spyke',
      #version = '',
      description = 'Multichannel spike viewer and sorter for Swindale Lab .srf files',
      author = 'Martin Spacek, Reza Lotun',
      #author_email = '',
      #url = '',
      #long_description = '',
      packages = ['spyke', 'spyke.gui', 'spyke.gui.res'])
