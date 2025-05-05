import subprocess

#subprocess.run([
#    "pip", "install", "/Users/Shared/Thermo-Calc/2025a/SDK/TC-Python/TC_Python-2025.1.35139-246-py3-none-any.whl"
#])


version = '2025a'

print('Testing TC-Python version: ' + version)
print('Please make sure that the variable "version" above, matches the release that you want to test, if not change it and re-run this script.')

# below this line, nothing needs to be manually updated.

import sys
print('')
print('Python version: (should be at least 3.5 and can NOT be older than 3.0)')
print(str(sys.version_info[0]) + '.' + str(sys.version_info[1]))
if sys.version_info[0] < 3 or sys.version_info[1] < 5:
    print('Wrong version of Python !!!!!')

print('')
print('Python executable path: (gives a hint about the used virtual / conda environment, in case of Anaconda the corresponding \n'
      'environment name can be found by running `conda env list` on the Anaconda command prompt, '
      'TC-Python must be installed into \nEACH separate environment used!)')
print(sys.executable)

import os

os.environ['TC25A_HOME'] = '/Applications/Thermo-Calc-2025a.app/Contents/Resources/'
os.environ['JAVA_HOME'] = '/Library/Java/JavaVirtualMachines/jdk-19.jdk/Contents/Home'
os.environ['LSHOST'] = '128.194.43.55'
os.environ['LSERVRC'] = 'coe-vtls3.engr.tamu.edu'
os.environ['TCLM_LICENSE_FILE'] = '5093@128.194.43.55'

print('')
print('Thermo-Calc ' + version + ' installation directory: (must be a valid path to a complete installation of ' + version + ')')
tc_env_variable = 'TC' + version[2:].upper() + '_HOME'
try:
    print(os.environ[tc_env_variable])
except:
    print('No Thermo-calc environment variable for ' + version + ' was found. (' + tc_env_variable + ')')

print('')
print('Url of license server: (if license server is NO-NET, you need a local license file)')
try:
    print(os.environ['LSHOST'])
except:
    print('No Thermo-calc license server url was found. (LSHOST)')


print('')
print('Path to local license file: (only necessary if not using license server)')
try:
    print(os.environ['LSERVRC'])
except:
    print('No path to local license file was found. (LSERVRC)')


import tc_python
numerical_version = version[:-1]
if version[-1] == 'a':
    numerical_version += '.1.*'
elif version[-1] == 'b':
    numerical_version += '.2.*'
print('')
print('TC-Python version: (needs to be ' + numerical_version + ')')
print(tc_python.__version__)


with tc_python.TCPython() as session:
    print('')
    print('Lists the databases: (should be a complete list of the installed databases that you have license for or do not require license)')
    print(session.get_databases())



"""
from tc_python import TCSession, TCOpticalProperties

# Start a Thermo-Calc session
session = TCSession()

# List the databases currently loaded
print("Loaded databases:", session.get_databases())

# Access the Optical Properties module (if available)
optical_properties = TCOpticalProperties(session)
print("Optical Properties module is available.")

# Close the session
session.close()"
"""