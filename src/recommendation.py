import sys as Sy
import numpy as np
import re


if __name__ == '__main__':
  name="Wang rui rui."
  print(re.split(r'\W+',name))