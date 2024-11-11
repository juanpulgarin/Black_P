import sys
import numpy as np
import h5py
#----------------------------------------------------------------------
def GetOrbData_tb(fs, fp, fd):
    ang = 1.0 / 0.5291772

    a1 = np.array([3.31067978852923, 0.0, 0.0]) * ang
    a2 = np.array([0.0, 4.58545824667620, 0.0]) * ang
    a3 = np.array([0.0, 0.0,  10.9888937548268 ]) * ang

    norb = 32
    nat = 8
    lkmax = 2

    Ls = []
    Ms = []
    
    for iat in range(nat):
        # s
        Ls.extend([0])
        Ms.extend([0])
        # p
        Ls.extend([1, 1, 1])
        Ms.extend([0, 1, -1])
    
    pos = np.zeros([norb,3])

    pos = np.array([
        [ 4.69220875,  3.55523135,  18.79104606],
        [ 4.69220775,  3.42178301,  18.57531568],
        [ 4.69220248,  3.83496996,  18.85641269],
        [ 4.69222478,  3.56355215,  18.83524904],
        [ 1.56406880,  0.77740610,  18.79103364],
        [ 1.56405419,  0.91087872,  18.57525496],
        [ 1.56405852,  0.49769379,  18.85644493],
        [ 1.56408040,  0.76906732,  18.83524898],
        [ 1.56406636,  5.11000269,  12.35795619],
        [ 1.56408628,  5.24339621,  12.57367890],
        [ 1.56407435,  4.83053072,  12.29255126],
        [ 1.56405508,  5.10183382,  12.31381634],
        [ 4.69221559,  7.88787713,  12.35796235],
        [ 4.69223196,  7.75438447,  12.57369218],
        [ 4.69219471,  8.16730286,  12.29250774],
        [ 4.69218264,  7.89600078,  12.31378257],
        [ 1.56407157,  3.55526557,   8.40804898],
        [ 1.56409236,  3.42187910,   8.19234377],
        [ 1.56403809,  3.83461793,   8.47342580],
        [ 1.56398200,  3.56341258,   8.45224480],
        [ 4.69220403,  0.77737605,   8.40805471],
        [ 4.69215240,  0.91078749,   8.19243824],
        [ 4.69212967,  0.49799979,   8.47342884],
        [ 4.69218630,  0.76928183,   8.45230663],
        [ 4.69221229,  5.11002040,   1.97494252],
        [ 4.69222308,  5.24344706,   2.19074061],
        [ 4.69222550,  4.83050469,   1.90960274],
        [ 4.69226764,  5.10164770,   1.93069043],
        [ 1.56407294,  7.88786800,   1.97493925],
        [ 1.56410803,  7.75447074,   2.19062431],
        [ 1.56413635,  8.16742082,   1.90964768],
        [ 1.56410196,  7.89626621,   1.93061956]
    ])

    flm = np.zeros([norb,lkmax+1])
    for iorb in range(norb):
        if Ls[iorb] == 0:
            flm[iorb,:] = np.array([0.0, fp, 0.0])
        else:
            flm[iorb,:] = np.array([fs, 0.0, fd])

    orb_data = {
        'norb': norb,
        'ls': Ls,
        'ms': Ms,
        'pos': pos,
        'flm': flm
    }

    return orb_data
#----------------------------------------------------------------------
if __name__ == "__main__":

    f = h5py.File("./data/blackp.h5", "r")
    coords = np.copy(f['coords'])
    f.close()

    for i in range(coords.shape[1]):
        print("[ {:13.6f}, {:13.6f}, {:13.6f} ]".format(coords[0,i],coords[1,i],coords[2,i]))
    



