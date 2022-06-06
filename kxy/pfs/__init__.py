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
	from .pfs_selector import *
	from .pfs_predictor import *
except:
	import logging
	logging.warn('Importing the PFS submodule failed: Principal Feature Selector might not be available.')