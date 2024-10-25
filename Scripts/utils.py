import os
from ezc3d import c3d

def create_No_empty_folders(path, start_No, end_No):
    """
    This function creates indexed folders in a specific directory (path) from a start No° and an end No°
    """
    for i in range(start_No, end_No+1):
        if not os.path.exists(path + str(i)):
            os.makedirs(path + str(i))


def c3d2trc(c3dfile, trcfile, up_dir='z'):
    '''
    Import c3d file and convert to .trc file for OpenSim.
    Note, OpenSim uses y-axis for vertical direction. Specify which direction is vertical
    with up_dir. If it is z, the data will be rotated to have y as the vertical direction.

    Parameters
    ----------
    c3dfile : string
        Path to input .c3d file.
    trcfile : string
        Path for .trc file to be written.
    up_dir : string, optional
        Specify the vertical direction. OpenSim assumes y is up. The default is 'z'.

    Returns
    -------
    None.

    TAKEN from Allison Clouthier's Github :
    https://github.com/aclouthier/knee-model-tools/blob/main/utils.py
    '''
    c = c3d(c3dfile)
    pts = c['data']['points']
    labels = c['parameters']['POINT']['LABELS']['value']

    freq = c['header']['points']['frame_rate']
    time = np.arange(start=c['header']['points']['first_frame'] / freq,
                     stop=(c['header']['points']['last_frame'] + 1) / freq, step=1 / freq)

    # check units
    if c['parameters']['POINT']['UNITS'] == 'm':
        pts = pts * 1000

    # rotate if y is not up direction
    if up_dir == 'z':
        dummy = pts.copy()
        dummy[1, :, :] = pts[2, :, :].copy()
        dummy[2, :, :] = -pts[1, :, :].copy()
        pts = dummy.copy()

    writeTRCfile(time, pts[:3, :, :], labels, trcfile)

path = "D:\Antoine\TN10_uOttawa\Data\Opensim\Kinematic_comparison\T0027_running2_markerless\Mocap"
c3d2trc(path + "\pose_0.c3d", path + "\pose_0.trc")