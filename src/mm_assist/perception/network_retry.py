"""Locate the registered native connection retry dialog without OCR."""
from functools import lru_cache
import hashlib
import json

import cv2
import numpy as np

from mm_assist.config import ROOT

ASSETS = ROOT/'knowledge/templates/network_retry'
WAIT_S = 5.0


def _gray(image):
    return cv2.GaussianBlur(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),(3,3),.5)


@lru_cache(maxsize=1)
def _registration():
    row=json.loads((ASSETS/'catalog.json').read_text('utf-8'))
    if (row['source_size'] != [736,552] or row['wait_s'] != WAIT_S
            or not .94 <= row['threshold'] <= 1 or len(row['anchors']) != 2):
        raise ValueError('invalid network retry registration')
    anchors=[]
    for anchor in row['anchors']:
        raw=(ASSETS/anchor['file']).read_bytes()
        image=cv2.imdecode(np.frombuffer(raw,np.uint8),1)
        x1,y1,x2,y2=anchor['crop_xyxy']
        if (hashlib.sha256(raw).hexdigest() != anchor['sha256'] or image is None
                or image.shape[:2] != (y2-y1,x2-x1)
                or not 0 <= x1 < x2 <= 736 or not 0 <= y1 < y2 <= 552):
            raise ValueError('invalid network retry anchor')
        anchors.append((_gray(image),x1,y1))
    x1,y1,x2,y2=row['anchors'][1]['crop_xyxy']
    if not x1 < row['click_xy'][0] < x2 or not y1 < row['click_xy'][1] < y2:
        raise ValueError('network retry point outside button')
    return row,anchors


def locate_network_retry(frame):
    """Both full title and green button must agree on their displacement."""
    if (not isinstance(frame,np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3
            or frame.dtype != np.uint8 or min(frame.shape[:2]) < 240
            or max(frame.shape[:2]) > 16384 or np.max(frame) < 5):
        return None
    row,anchors=_registration()
    scale=frame.shape[0]/552
    image=_gray(cv2.resize(frame,(round(frame.shape[1]/scale),552),interpolation=cv2.INTER_AREA))
    offsets=[]
    for template,x,y in anchors:
        x += image.shape[1]/2-368
        th,tw=template.shape
        left,top=max(0,round(x)-12),max(0,y-12)
        right,bottom=min(image.shape[1],round(x)+tw+13),min(552,y+th+13)
        region=image[top:bottom,left:right]
        if region.shape[0]<th or region.shape[1]<tw:
            return None
        _,score,_,point=cv2.minMaxLoc(cv2.matchTemplate(region,template,cv2.TM_CCOEFF_NORMED))
        if not np.isfinite(score) or score < row['threshold']:
            return None
        offsets.append((left+point[0]-x,top+point[1]-y))
    if max(abs(offsets[0][i]-offsets[1][i]) for i in (0,1))>3:
        return None
    return (round((row['click_xy'][0]+image.shape[1]/2-368+offsets[1][0])*scale),
            round((row['click_xy'][1]+offsets[1][1])*scale))
