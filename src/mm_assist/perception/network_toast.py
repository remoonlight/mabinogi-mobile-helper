"""Read-only recognition of the registered top-center network warning."""
from functools import lru_cache
import hashlib
import json

import cv2
import numpy as np

from mm_assist.config import ROOT
from mm_assist.platform.match import _peak_and_margin

ASSETS = ROOT / 'knowledge/templates/network'
POLL_INTERVAL_S = 2.0


@lru_cache(maxsize=1)
def _registration():
    row = json.loads((ASSETS / 'catalog.json').read_text('utf-8'))
    raw = (ASSETS / row['template']).read_bytes()
    if (row['source_size'] != [1919,1440] or row['poll_interval_s'] != POLL_INTERVAL_S
            or not .94 <= row['threshold'] <= 1 or not .02 <= row['min_margin'] <= 1
            or hashlib.sha256(raw).hexdigest() != row['template_sha256']):
        raise ValueError('invalid network warning registration')
    image = cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
    x,y,w,h = row['crop_xywh']
    bounds = row['search_xyxy']
    if (image is None or image.shape[:2] != (h,w) or len(bounds) != 4
            or not 0 <= bounds[0] <= x < x+w <= bounds[2] <= 1919
            or not 0 <= bounds[1] <= y < y+h <= bounds[3] <= 1440):
        raise ValueError('invalid network warning crop/region')
    image = cv2.resize(image,None,fx=.5,fy=.5,interpolation=cv2.INTER_AREA)
    return row, _gray(image)


def _gray(image):
    return cv2.GaussianBlur(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),(3,3),.5)


def network_unstable(frame) -> bool:
    """Only the full caption in the registered region establishes presence.

    Invalid/black/clipped pixels raise; they never establish disappearance.
    """
    if (not isinstance(frame,np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3
            or frame.dtype != np.uint8 or min(frame.shape[:2]) < 240
            or max(frame.shape[:2]) > 16384 or np.max(frame) < 5):
        raise ValueError('network warning frame unavailable')
    row, template = _registration()
    fh,fw = frame.shape[:2]
    scale = fh/1440
    x1,y1,x2,y2 = row['search_xyxy']
    left = round(fw/2 + (x1-1919/2)*scale)
    right = round(fw/2 + (x2-1919/2)*scale)
    top,bottom = round(y1*scale),round(y2*scale)
    if not 0 <= left < right <= fw or not 0 <= top < bottom <= fh:
        raise ValueError('network warning region unavailable')
    region = frame[top:bottom,left:right]
    region = cv2.resize(region,None,fx=720/fh,fy=720/fh,interpolation=cv2.INTER_AREA)
    region = _gray(region)
    th,tw = template.shape
    if region.shape[0] < th or region.shape[1] < tw:
        raise ValueError('network warning region too small')
    result = cv2.matchTemplate(region,template,cv2.TM_CCOEFF_NORMED)
    score,_,margin = _peak_and_margin(result,tw,th,True)
    return bool(np.isfinite(score) and score >= row['threshold'] and margin >= row['min_margin'])
