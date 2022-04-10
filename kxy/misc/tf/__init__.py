#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (C) 2022 KXY TECHNOLOGIES, INC.
Author: Dr Yves-Laurent Kom Samo

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
try:
	from pkg_resources import parse_version
	import tensorflow as tf
	assert parse_version(tf.__version__) >= parse_version('2.4.1')
except:
	import logging
	logging.warning('You need tensorflow version 2.8 or higher to estimate mutual information or copula entropy locally.')

from .generators import *
from .ops import *
from .config import *
from .initializers import *
from .layers import *
from .losses import *
from .models import *
from .learners import *