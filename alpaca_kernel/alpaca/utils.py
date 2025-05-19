import base64
import urllib
from io import BytesIO

import numpy as np


def _to_png(fig):
    """Return a not base64-encoded PNG from a
    matplotlib figure."""
    imgdata = BytesIO()
    fig.savefig(imgdata, format='png',bbox_inches='tight')
    imgdata.seek(0)
    return imgdata.read()


def string_to_numpy(string):
    line_Items = []
    width = None
    for line in string.split("],"):
        line_Parts = line.split()
        n = len(line_Parts)
        if n == 0:
            continue
        if width is None:
            width = n
        else:
            assert n == width, "Invalid Array"
        line = line.split("[")[-1].split("]")[0]

        line_Items.append(np.fromstring(line, dtype=float, sep=','))
    return np.array(line_Items)


def string_is_array(string):
    if string.count('[') != string.count(']'):
        return False
    if sum(cc.isalpha() for cc in string) > 0:  # cant contain alphanumerics
        return False

    number_of_numbers = 0
    number_flag = False
    for cc in string:
        if cc.isnumeric() and not number_flag:  # recognize start of number
            number_flag = True
        if number_flag and cc in [']', ',']:  # recognize end of number
            number_flag = False
            number_of_numbers += 1

    if number_of_numbers != string.count(',') + 1:
        return False

    return True


def unpack_Thonny_string(output):
    ii_label_start = 0
    ii_number_start = 0
    ii_number_end = 0
    points = {}

    number_flag = False
    for ii, cc in enumerate(output):
        # Previous end is new start
        if cc.isnumeric() and output[ii - 1] == ' ' and not number_flag:  # recognize start of number
            number_flag = True
            ii_number_start = ii

        at_end = ii == len(output) - 1
        if number_flag and (cc in [' '] or at_end):  # recognize end of number
            ii_number_end = ii

            if at_end:
                ii_number_end = ii + 1


            label = output[ii_label_start:ii_number_start].split(':')[0]
            label = label.rstrip()
            number = output[ii_number_start:ii_number_end]
            points[label] = float(number)

            # Prep for new loop
            number_flag = False
            ii_label_start = ii_number_end + 1

    return points
