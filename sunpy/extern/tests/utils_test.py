from __future__ import absolute_import, division, print_function, unicode_literals
from ...wcs import WCS
from .. import utils
from ...tests.helper import pytest
import numpy as np

def test_wcs_to_celestial_frame():

    from astropy.coordinates.builtin_frames import ICRS, FK5, FK4, Galactic
    from astropy.time import Time

    mywcs = WCS(naxis=2)
    with pytest.raises(ValueError) as exc:
        assert utils.wcs_to_celestial_frame(mywcs) is None
    assert exc.value.args[0] == "Could not determine celestial frame corresponding to the specified WCS object"

    mywcs.wcs.ctype = ['XOFFSET', 'YOFFSET']
    with pytest.raises(ValueError):
        assert utils.wcs_to_celestial_frame(mywcs) is None

    mywcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    frame = utils.wcs_to_celestial_frame(mywcs)
    assert isinstance(frame, ICRS)

    mywcs.wcs.equinox = 1987.
    frame = utils.wcs_to_celestial_frame(mywcs)
    assert isinstance(frame, FK5)
    assert frame.equinox == Time(1987., format='jyear')

    mywcs.wcs.equinox = 1982
    frame = utils.wcs_to_celestial_frame(mywcs)
    assert isinstance(frame, FK4)
    assert frame.equinox == Time(1982., format='byear')

    mywcs.wcs.equinox = np.nan
    mywcs.wcs.ctype = ['GLON-SIN', 'GLAT-SIN']
    frame = utils.wcs_to_celestial_frame(mywcs)
    assert isinstance(frame, Galactic)

    mywcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    mywcs.wcs.radesys = 'ICRS'

    for equinox in [np.nan, 1987, 1982]:
        mywcs.wcs.equinox = equinox
        frame = utils.wcs_to_celestial_frame(mywcs)
        assert isinstance(frame, ICRS)

    # Flipped order
    mywcs = WCS(naxis=2)
    mywcs.wcs.ctype = ['DEC--TAN', 'RA---TAN']
    assert isinstance(frame, ICRS)

    # More than two dimensions
    mywcs = WCS(naxis=3)
    mywcs.wcs.ctype = ['DEC--TAN', 'VELOCITY', 'RA---TAN']
    assert isinstance(frame, ICRS)


def test_wcs_to_celestial_frame_extend():

    from ...coordinates.builtin_frames import ICRS, FK5, FK4, Galactic
    from ...time import Time

    mywcs = WCS(naxis=2)
    mywcs.wcs.ctype = ['XOFFSET', 'YOFFSET']
    with pytest.raises(ValueError):
        utils.wcs_to_celestial_frame(mywcs)

    class OffsetFrame(object):
        pass

    def identify_offset(wcs):
        if wcs.wcs.ctype[0].endswith('OFFSET') and wcs.wcs.ctype[1].endswith('OFFSET'):
            return OffsetFrame()

    from ..utils import custom_frame_mappings

    with custom_frame_mappings(identify_offset):
        frame = utils.wcs_to_celestial_frame(mywcs)
    assert isinstance(frame, OffsetFrame)

    # Check that things are back to normal after the context manager
    with pytest.raises(ValueError):
        utils.wcs_to_celestial_frame(mywcs)
