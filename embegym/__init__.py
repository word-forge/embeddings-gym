import pyximport
import numpy

pyximport.install(setup_args={'include_dirs': numpy.get_include()})

# CYTHON_TRACE=1
# setup_args={
#     'options': {
#         'build_ext': {
#             'cython_directives': {
#                 'language_level': 3,
#                 'optimize.use_switch': True,
#                 'profile': True,
#             }
#         }
#     }
# }


from . import data, evaluation, models, utils
