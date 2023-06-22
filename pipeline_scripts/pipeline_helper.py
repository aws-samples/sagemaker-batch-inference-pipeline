import logging
from __main__ import __file__ as mf
from os import path
import yaml

# Source files
tfvarsfile = './env_files/dev_env.tfvars'
parameterspath = './pipeline_scripts/'
sourcename = path.splitext(path.basename(mf))[0]
parametersfile = parameterspath + sourcename + ".yaml"

# Setup logger
LOGGER = logging.getLogger()
LOGGER.setLevel(logging.INFO)

def log(msg):
    LOGGER.info("Log: %s", msg)

def file_exists(filepath,msg):
    try:
        open(filepath)
    except IOError:
        raise IOError('Could not access file',filepath,msg)
    except FileNotFoundError:
        raise FileNotFoundError('Could not find file',filepath,msg)

def get_parameter(key:str,default=None,basekey="Parameters"):
    with open(parametersfile) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)
    file.close
    param = params.get(basekey).get(key)
    if param == None:
        if default == None:
            raise ValueError("Could not retreive a value and no default specified")
        else:
            return(default)
    else:
        return(param)

def get_variable(varname:str):
    return(get_var_tfvars(varname, tfvarsfile))

def printvars(currentvars):
    for k,v in currentvars.items():
        if not k.startswith('_'):
            print(k,' : ',v,' type:' , type(v))

def get_var_tfvars(varname:str,tfvarfilename):
    with open(tfvarfilename) as file:
        for line in file:
            if not line[0] == '#':
                srcline = line.rstrip().casefold()
                varname = varname.casefold()
                if srcline.find(varname) >= 0:
                    return line.rstrip().split("\"")[1]

# Test for vars file
file_exists(tfvarsfile,"Testing for tfvars file")
log("Pipeline Helper Loaded")