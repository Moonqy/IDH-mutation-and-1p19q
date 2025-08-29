from __future__ import annotations
import random
import re
from collections import OrderedDict
from collections.abc import Callable, Sequence

import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url

from monai.networks.layers.factories import Conv, Dropout, Pool
from monai.networks.layers.utils import get_act_layer, get_norm_layer
from monai.utils.module import look_up_option

#The code will be made public after the journal article is accepted.