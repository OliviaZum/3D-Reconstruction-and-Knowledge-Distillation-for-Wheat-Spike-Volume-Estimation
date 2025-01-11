from pathlib import Path
import shutil
import subprocess
import json
import numpy as np
import struct
import argparse
import tempfile

def get_sfm_base():
    return """
{
    "sfm_data_version": "0.3",
    "root_path": "./build/images",
    "views": [
        {
            "key": 0,
            "value": {
                "polymorphic_id": 1073741824,
                "ptr_wrapper": {
                    "id": 2147483649,
                    "data": {
                        "local_path": "",
                        "filename": "cam_01.png",
                        "width": 4096,
                        "height": 3000,
                        "id_view": 0,
                        "id_intrinsic": 0,
                        "id_pose": 0
                    }
                }
            }
        },
        {
            "key": 1,
            "value": {
                "polymorphic_id": 1073741824,
                "ptr_wrapper": {
                    "id": 2147483650,
                    "data": {
                        "local_path": "",
                        "filename": "cam_02.png",
                        "width": 4096,
                        "height": 3000,
                        "id_view": 1,
                        "id_intrinsic": 1,
                        "id_pose": 1
                    }
                }
            }
        },
        {
            "key": 2,
            "value": {
                "polymorphic_id": 1073741824,
                "ptr_wrapper": {
                    "id": 2147483651,
                    "data": {
                        "local_path": "",
                        "filename": "cam_03.png",
                        "width": 4096,
                        "height": 3000,
                        "id_view": 2,
                        "id_intrinsic": 2,
                        "id_pose": 2
                    }
                }
            }
        },
        {
            "key": 3,
            "value": {
                "polymorphic_id": 1073741824,
                "ptr_wrapper": {
                    "id": 2147483652,
                    "data": {
                        "local_path": "",
                        "filename": "cam_04.png",
                        "width": 4096,
                        "height": 3000,
                        "id_view": 3,
                        "id_intrinsic": 3,
                        "id_pose": 3
                    }
                }
            }
        },
        {
            "key": 4,
            "value": {
                "polymorphic_id": 1073741824,
                "ptr_wrapper": {
                    "id": 2147483653,
                    "data": {
                        "local_path": "",
                        "filename": "cam_05.png",
                        "width": 4096,
                        "height": 3000,
                        "id_view": 4,
                        "id_intrinsic": 4,
                        "id_pose": 4
                    }
                }
            }
        },
        {
            "key": 5,
            "value": {
                "polymorphic_id": 1073741824,
                "ptr_wrapper": {
                    "id": 2147483654,
                    "data": {
                        "local_path": "",
                        "filename": "cam_06.png",
                        "width": 4096,
                        "height": 3000,
                        "id_view": 5,
                        "id_intrinsic": 5,
                        "id_pose": 5
                    }
                }
            }
        },
        {
            "key": 6,
            "value": {
                "polymorphic_id": 1073741824,
                "ptr_wrapper": {
                    "id": 2147483655,
                    "data": {
                        "local_path": "",
                        "filename": "cam_07.png",
                        "width": 4096,
                        "height": 3000,
                        "id_view": 6,
                        "id_intrinsic": 6,
                        "id_pose": 6
                    }
                }
            }
        },
        {
            "key": 7,
            "value": {
                "polymorphic_id": 1073741824,
                "ptr_wrapper": {
                    "id": 2147483656,
                    "data": {
                        "local_path": "",
                        "filename": "cam_08.png",
                        "width": 4096,
                        "height": 3000,
                        "id_view": 7,
                        "id_intrinsic": 7,
                        "id_pose": 7
                    }
                }
            }
        },
        {
            "key": 8,
            "value": {
                "polymorphic_id": 1073741824,
                "ptr_wrapper": {
                    "id": 2147483657,
                    "data": {
                        "local_path": "",
                        "filename": "cam_09.png",
                        "width": 4096,
                        "height": 3000,
                        "id_view": 8,
                        "id_intrinsic": 8,
                        "id_pose": 8
                    }
                }
            }
        },
        {
            "key": 9,
            "value": {
                "polymorphic_id": 1073741824,
                "ptr_wrapper": {
                    "id": 2147483658,
                    "data": {
                        "local_path": "",
                        "filename": "cam_10.png",
                        "width": 4096,
                        "height": 3000,
                        "id_view": 9,
                        "id_intrinsic": 9,
                        "id_pose": 9
                    }
                }
            }
        },
        {
            "key": 10,
            "value": {
                "polymorphic_id": 1073741824,
                "ptr_wrapper": {
                    "id": 2147483659,
                    "data": {
                        "local_path": "",
                        "filename": "cam_11.png",
                        "width": 4096,
                        "height": 3000,
                        "id_view": 10,
                        "id_intrinsic": 10,
                        "id_pose": 10
                    }
                }
            }
        },
        {
            "key": 11,
            "value": {
                "polymorphic_id": 1073741824,
                "ptr_wrapper": {
                    "id": 2147483660,
                    "data": {
                        "local_path": "",
                        "filename": "cam_12.png",
                        "width": 4096,
                        "height": 3000,
                        "id_view": 11,
                        "id_intrinsic": 11,
                        "id_pose": 11
                    }
                }
            }
        }
    ],
    "intrinsics": [
        {
            "key": 0,
            "value": {
                "polymorphic_id": 2147483649,
                "polymorphic_name": "pinhole_radial_k3",
                "ptr_wrapper": {
                    "id": 2147483661,
                    "data": {
                        "width": 4096,
                        "height": 3000,
                        "focal_length": 10031.850513622649,
                        "principal_point": [
                            2053.3214454814776,
                            1587.318070675283
                        ],
                        "disto_k3": [
                            0.01165982686491528,
                            -0.576203381465697,
                            10.563453582163103
                        ]
                    }
                }
            }
        },
        {
            "key": 1,
            "value": {
                "polymorphic_id": 1,
                "ptr_wrapper": {
                    "id": 2147483662,
                    "data": {
                        "width": 4096,
                        "height": 3000,
                        "focal_length": 10081.518922051722,
                        "principal_point": [
                            2072.8854607234309,
                            1548.370933353022
                        ],
                        "disto_k3": [
                            -0.005386377716238078,
                            0.21421550258329043,
                            -2.5475149644987895
                        ]
                    }
                }
            }
        },
        {
            "key": 2,
            "value": {
                "polymorphic_id": 1,
                "ptr_wrapper": {
                    "id": 2147483663,
                    "data": {
                        "width": 4096,
                        "height": 3000,
                        "focal_length": 10057.544286641785,
                        "principal_point": [
                            2043.7962112536575,
                            1577.9495579342625
                        ],
                        "disto_k3": [
                            0.005176982600955843,
                            -0.510725068987643,
                            10.152137087879286
                        ]
                    }
                }
            }
        },
        {
            "key": 3,
            "value": {
                "polymorphic_id": 1,
                "ptr_wrapper": {
                    "id": 2147483664,
                    "data": {
                        "width": 4096,
                        "height": 3000,
                        "focal_length": 10083.8099225929,
                        "principal_point": [
                            2079.99060920274,
                            1554.2192374846389
                        ],
                        "disto_k3": [
                            -0.044776832098326179,
                            1.712289620724554,
                            -20.2983897496404
                        ]
                    }
                }
            }
        },
        {
            "key": 4,
            "value": {
                "polymorphic_id": 1,
                "ptr_wrapper": {
                    "id": 2147483665,
                    "data": {
                        "width": 4096,
                        "height": 3000,
                        "focal_length": 10054.909286080187,
                        "principal_point": [
                            2012.3313265060402,
                            1523.476306979742
                        ],
                        "disto_k3": [
                            -0.02773994745605705,
                            0.9136297482973947,
                            -5.892075117832146
                        ]
                    }
                }
            }
        },
        {
            "key": 5,
            "value": {
                "polymorphic_id": 1,
                "ptr_wrapper": {
                    "id": 2147483666,
                    "data": {
                        "width": 4096,
                        "height": 3000,
                        "focal_length": 10058.180208922438,
                        "principal_point": [
                            2058.9412112015936,
                            1489.888888289101
                        ],
                        "disto_k3": [
                            -0.029094947935103214,
                            1.1671550628959154,
                            -12.836452848032338
                        ]
                    }
                }
            }
        },
        {
            "key": 6,
            "value": {
                "polymorphic_id": 1,
                "ptr_wrapper": {
                    "id": 2147483667,
                    "data": {
                        "width": 4096,
                        "height": 3000,
                        "focal_length": 10046.517947978173,
                        "principal_point": [
                            2058.2550784028246,
                            1505.3068017347203
                        ],
                        "disto_k3": [
                            -0.02401794378654959,
                            0.651348804677165,
                            -2.8554632476062755
                        ]
                    }
                }
            }
        },
        {
            "key": 7,
            "value": {
                "polymorphic_id": 1,
                "ptr_wrapper": {
                    "id": 2147483668,
                    "data": {
                        "width": 4096,
                        "height": 3000,
                        "focal_length": 10061.479387943476,
                        "principal_point": [
                            2046.7631372246543,
                            1521.6982489889986
                        ],
                        "disto_k3": [
                            -0.03177752461208436,
                            0.6195399372395596,
                            -1.8407518133264729
                        ]
                    }
                }
            }
        },
        {
            "key": 8,
            "value": {
                "polymorphic_id": 1,
                "ptr_wrapper": {
                    "id": 2147483669,
                    "data": {
                        "width": 4096,
                        "height": 3000,
                        "focal_length": 10029.433065347299,
                        "principal_point": [
                            2044.1933310312077,
                            1460.6839955756466
                        ],
                        "disto_k3": [
                            -0.04860485728842088,
                            1.7949156495623536,
                            -19.150016519017567
                        ]
                    }
                }
            }
        },
        {
            "key": 9,
            "value": {
                "polymorphic_id": 1,
                "ptr_wrapper": {
                    "id": 2147483670,
                    "data": {
                        "width": 4096,
                        "height": 3000,
                        "focal_length": 10079.826840814665,
                        "principal_point": [
                            2057.05743301195,
                            1500.8729929364693
                        ],
                        "disto_k3": [
                            -0.021649805908673836,
                            1.0750080550823416,
                            -11.351554540848238
                        ]
                    }
                }
            }
        },
        {
            "key": 10,
            "value": {
                "polymorphic_id": 1,
                "ptr_wrapper": {
                    "id": 2147483671,
                    "data": {
                        "width": 4096,
                        "height": 3000,
                        "focal_length": 10074.165524219536,
                        "principal_point": [
                            2054.292474883853,
                            1493.5465177822737
                        ],
                        "disto_k3": [
                            -0.00858844465803132,
                            0.012459285965910552,
                            2.86164954699394
                        ]
                    }
                }
            }
        },
        {
            "key": 11,
            "value": {
                "polymorphic_id": 1,
                "ptr_wrapper": {
                    "id": 2147483672,
                    "data": {
                        "width": 4096,
                        "height": 3000,
                        "focal_length": 10068.447995482587,
                        "principal_point": [
                            2030.6966812925185,
                            1500.0280023541182
                        ],
                        "disto_k3": [
                            -0.007415382076092796,
                            0.1763356163596315,
                            1.3840134142337089
                        ]
                    }
                }
            }
        }
    ],
    "extrinsics": [
        {
            "key": 3,
            "value": {
                "rotation": [
                    [
                        0.9999976099420017,
                        -0.0015283180156259078,
                        0.0015634431001031694
                    ],
                    [
                        0.001526210845709413,
                        0.9999979266760134,
                        0.0013480816477220207
                    ],
                    [
                        -0.001565500156047862,
                        -0.0013456922819126395,
                        0.9999978691585016
                    ]
                ],
                "center": [
                    0.011295228485298719,
                    0.014915963152828477,
                    -0.014438395022718515
                ]
            }
        },
        {
            "key": 11,
            "value": {
                "rotation": [
                    [
                        0.9921145002307514,
                        -0.006258611481698739,
                        -0.1251784654571521
                    ],
                    [
                        -0.02025978318465925,
                        0.9776096663764149,
                        -0.20944899472832369
                    ],
                    [
                        0.123686537736315,
                        0.2103334732982744,
                        0.9697739274661323
                    ]
                ],
                "center": [
                    -0.2304095635916308,
                    -0.4988361835831866,
                    0.31386280312780359
                ]
            }
        },
        {
            "key": 8,
            "value": {
                "rotation": [
                    [
                        0.9749711458691984,
                        -0.03715076771756099,
                        -0.21920557743930264
                    ],
                    [
                        -0.03404477707747786,
                        0.9493665902645955,
                        -0.3123203970974687
                    ],
                    [
                        0.21970939414655694,
                        0.311966180454503,
                        0.9243402427550049
                    ]
                ],
                "center": [
                    -0.5244662949436505,
                    -0.8447702730725603,
                    0.19834460879864325
                ]
            }
        },
        {
            "key": 5,
            "value": {
                "rotation": [
                    [
                        0.9997837835855064,
                        -0.0011039190340080426,
                        -0.0207645718042951
                    ],
                    [
                        -0.0009098299332312683,
                        0.9953110896678499,
                        -0.09672128511186032
                    ],
                    [
                        0.020773981056647969,
                        0.09671926461136649,
                        0.9950948826941551
                    ]
                ],
                "center": [
                    0.012182512357503173,
                    -0.28995444724856098,
                    -0.0012880160123784554
                ]
            }
        },
        {
            "key": 10,
            "value": {
                "rotation": [
                    [
                        0.9908071236541428,
                        0.002476564215710301,
                        -0.13525941869566616
                    ],
                    [
                        -0.002199676198817349,
                        0.9999951683930638,
                        0.002196500659767223
                    ],
                    [
                        0.13526420495025438,
                        -0.0018787815768376614,
                        0.9908077841029319
                    ]
                ],
                "center": [
                    -0.2365673982532484,
                    -0.021138005071923005,
                    0.26340616622338555
                ]
            }
        },
        {
            "key": 6,
            "value": {
                "rotation": [
                    [
                        0.9708082574637409,
                        -0.04336082571613799,
                        -0.23590499365937463
                    ],
                    [
                        -0.01052892924847804,
                        0.9748719975945521,
                        -0.22251680825251833
                    ],
                    [
                        0.23962568495279347,
                        0.2185049818636229,
                        0.9459572421688426
                    ]
                ],
                "center": [
                    -0.5325954530271813,
                    -0.577885713363203,
                    0.12074805274039796
                ]
            }
        },
        {
            "key": 4,
            "value": {
                "rotation": [
                    [
                        0.9735493410260201,
                        -0.018681332618521237,
                        -0.2277118538842417
                    ],
                    [
                        -0.005971525391663879,
                        0.9942307042317363,
                        -0.1070964408715917
                    ],
                    [
                        0.22839882108342586,
                        0.10562345655422284,
                        0.9678210908805612
                    ]
                ],
                "center": [
                    -0.5398775325569368,
                    -0.2880581757009142,
                    0.06918641077064565
                ]
            }
        },
        {
            "key": 7,
            "value": {
                "rotation": [
                    [
                        0.9983018391997294,
                        0.05149091626349427,
                        -0.0272419417953903
                    ],
                    [
                        -0.0561089236266211,
                        0.9756461753121691,
                        -0.21205265687606504
                    ],
                    [
                        0.01565971072209231,
                        0.21322107339820327,
                        0.9768784711104132
                    ]
                ],
                "center": [
                    0.016550748840968855,
                    -0.5805436550558398,
                    0.047465613168510199
                ]
            }
        },
        {
            "key": 2,
            "value": {
                "rotation": [
                    [
                        0.9716567532976491,
                        0.00024705179934015394,
                        -0.23639605059408229
                    ],
                    [
                        0.0009461053326885954,
                        0.9999873810153284,
                        0.004933831655410376
                    ],
                    [
                        0.2363942864279315,
                        -0.005017646411706795,
                        0.9716442582389483
                    ]
                ],
                "center": [
                    -0.5392884418483217,
                    0.013003888417626196,
                    0.053636163054631928
                ]
            }
        },
        {
            "key": 1,
            "value": {
                "rotation": [
                    [
                        0.9991193481446432,
                        -0.03025371092264262,
                        -0.029073031118761626
                    ],
                    [
                        0.03298755634934579,
                        0.9945720204285935,
                        0.09868291294182705
                    ],
                    [
                        0.025929698978623795,
                        -0.09955505590372924,
                        0.9946941447273545
                    ]
                ],
                "center": [
                    0.014048952599435744,
                    0.2892522761176684,
                    -0.008012689586596067
                ]
            }
        },
        {
            "key": 0,
            "value": {
                "rotation": [
                    [
                        0.9722110886910516,
                        -0.0005170046807770612,
                        -0.23410538595325026
                    ],
                    [
                        0.023667117985291146,
                        0.9950913320386845,
                        0.09608906507895095
                    ],
                    [
                        0.2329065618492328,
                        -0.0989594543620599,
                        0.9674510632791371
                    ]
                ],
                "center": [
                    -0.5348261660794089,
                    0.2859786721363806,
                    0.06751234269076298
                ]
            }
        },
        {
            "key": 9,
            "value": {
                "rotation": [
                    [
                        0.9993827319447615,
                        0.03261733178898107,
                        -0.01304855385062632
                    ],
                    [
                        -0.03508550434110471,
                        0.9455001624613794,
                        -0.3237258873964754
                    ],
                    [
                        0.0017783351077600742,
                        0.3239838768403031,
                        0.9460609309509466
                    ]
                ],
                "center": [
                    0.02560972385393936,
                    -0.8448173503476906,
                    0.12271126915493394
                ]
            }
        }
    ],
    "structure": [],
    "control_points": []
}
    """

class HeaderDepthDataRaw:
    HAS_DEPTH = 1 << 0
    HAS_NORMAL = 1 << 1
    HAS_CONF = 1 << 2
    HAS_VIEWS = 1 << 3

    def __init__(self, data):
        # Unpack the fixed-size portion of the header
        unpacked = struct.unpack('<H B B I I I I f f', data)
        self.name = unpacked[0]
        self.type = unpacked[1]
        self.padding = unpacked[2]
        self.image_width = unpacked[3]
        self.image_height = unpacked[4]
        self.depth_width = unpacked[5]
        self.depth_height = unpacked[6]
        self.d_min = unpacked[7]
        self.d_max = unpacked[8]

    @staticmethod
    def expected_name():
        return int.from_bytes(b'DR', byteorder='little')

def read_depth_data(file_path):
    with open(file_path, 'rb') as f:
        header_data = f.read(28)  # Size of HeaderDepthDataRaw struct
        header = HeaderDepthDataRaw(header_data)

        assert header.name == HeaderDepthDataRaw.expected_name()
        assert header.type & HeaderDepthDataRaw.HAS_DEPTH
        assert header.depth_width > 0 and header.depth_height > 0
        assert header.image_width >= header.depth_width
        assert header.image_height >= header.depth_height

        # Read image file name
        file_name_size = struct.unpack('<H', f.read(2))[0]
        image_file_name = f.read(file_name_size).decode('utf-8')

        # Read neighbor IDs
        n_ids = struct.unpack('<I', f.read(4))[0]
        assert 0 < n_ids < 256
        ids = np.frombuffer(f.read(4 * n_ids), dtype=np.uint32)

        # Read pose
        K = np.frombuffer(f.read(72), dtype=np.float64).reshape(3, 3)
        R = np.frombuffer(f.read(72), dtype=np.float64).reshape(3, 3)
        C = np.frombuffer(f.read(24), dtype=np.float64)

        # Read depth-map
        d_min, d_max = header.d_min, header.d_max
        if header.type & HeaderDepthDataRaw.HAS_DEPTH:
            depth_map = np.frombuffer(
                f.read(4 * header.depth_width * header.depth_height), dtype=np.float32
            ).reshape(header.depth_height, header.depth_width)
        else:
            f.seek(4 * header.depth_width * header.depth_height, 1)
            depth_map = None

        # Read normal-map
        if header.type & HeaderDepthDataRaw.HAS_NORMAL:
            normal_map = np.frombuffer(
                f.read(12 * header.depth_width * header.depth_height), dtype=np.float32
            ).reshape(header.depth_height, header.depth_width, 3)
        else:
            f.seek(12 * header.depth_width * header.depth_height, 1)
            normal_map = None

        # Read confidence-map
        if header.type & HeaderDepthDataRaw.HAS_CONF:
            conf_map = np.frombuffer(
                f.read(4 * header.depth_width * header.depth_height), dtype=np.float32
            ).reshape(header.depth_height, header.depth_width)
        else:
            f.seek(4 * header.depth_width * header.depth_height, 1)
            conf_map = None

        # Read visibility-map
        if header.type & HeaderDepthDataRaw.HAS_VIEWS:
            views_map = np.frombuffer(
                f.read(4 * header.depth_width * header.depth_height), dtype=np.uint8
            ).reshape(header.depth_height, header.depth_width, 4)
        else:
            views_map = None

    return {
        "image_file_name": image_file_name,
        "ids": ids,
        "K": K,
        "R": R,
        "C": C,
        "depth_map": depth_map,
        "normal_map": normal_map,
        "conf_map": conf_map,
        "views_map": views_map,
        "d_min": d_min,
        "d_max": d_max
    }

def reconstruct_3d(openmvg_path: Path,
                   openmvs_path: Path,
                   out_dir: Path,
                   img_dir: Path,
                   scaled_calibration_path: Path,
                   mvs_size = 2500):
    
    
    out_dir.mkdir(parents=True, exist_ok=False)

    # Create an sfm data file from scaled configuration (For simplicity just adapt an existing sfm file)
    with tempfile.TemporaryDirectory(dir=r"D:\temp") as temp_dir:
        temp_dir = Path(temp_dir)
        with open(scaled_calibration_path) as f:
            calibration = json.load(f)
        sfm_base = json.loads(get_sfm_base())

        sfm_base["root_path"] = str(img_dir)
        pose_to_name = {}
        intrinsic_to_name = {}
        for view in sfm_base["views"]:
            d = view["value"]["ptr_wrapper"]["data"]
            pose_to_name[d["id_pose"]] = d["filename"]
            intrinsic_to_name[d["id_intrinsic"]] = d["filename"]
        for i in range(len(sfm_base["intrinsics"])):
            name = intrinsic_to_name[sfm_base["intrinsics"][i]["key"]]
            calib_intrinsic = calibration[name]["intrinsics"]
            sfm_base["intrinsics"][i]["value"]["ptr_wrapper"]["data"] = calib_intrinsic
        for i in range(len(sfm_base["extrinsics"])):
            name = pose_to_name[sfm_base["extrinsics"][i]["key"]]
            calib_extrinsic = calibration[name]["extrinsics"]
            sfm_base["extrinsics"][i]["value"] = calib_extrinsic
        
        with open(temp_dir / "sfm_data.json", "w") as f:
            json.dump(sfm_base, f, indent=4)

        # Export to openmvs and perform reconstruction
        (temp_dir / "mvs/images").mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [openmvg_path / "openMVG_main_openMVG2openMVS", "-i", temp_dir / "sfm_data.json", "-o", temp_dir / "mvs/mvs.mvs", "-d", temp_dir / "mvs/images"],
            check=True
        )

        # "--fusion-mode", "1"
        subprocess.run(
            [openmvs_path / "DensifyPointCloud", "-w", temp_dir / "mvs", "-i", "mvs.mvs", "--max-resolution", str(mvs_size), "--resolution-level", "0", "--estimate-roi", "0"], 
            check=True
        )

        # Collect and write depths and pointcloud
        mvs_path = temp_dir / "mvs"
        depths = {}
        for file in mvs_path.iterdir():
            if file.suffix == ".dmap":
                data = read_depth_data(file)
                depths[f"{data['image_file_name']}_depth"] = data["depth_map"]
        np.savez_compressed(out_dir / f"depths", **depths)
        shutil.move(mvs_path / "mvs_dense.ply", out_dir)

        """
        # Cleanup folder (It is so verbose to absolutely avoid deleting the wrong stuff)
        for file in temp_dir.rglob("*"):
            if file.is_file() and (file.name == "sfm_data.json" or file.suffix in [".log", ".dmap", ".mvs"]):
                file.unlink()
        
        for folder_name in ["mvs/images", "images"]:
            folder = temp_dir / folder_name
            if folder.exists() and folder.is_dir():
                for file in folder.rglob("*"):
                    if file.suffix in [".jpg", ".png"]:
                        file.unlink()
        (temp_dir / "mvs/images").rmdir()
        (temp_dir / "mvs").rmdir()
        """

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D Reconstruction with OpenMVG and OpenMVS")
    parser.add_argument("openmvg_path", type=Path, help="Path to the OpenMVG binary folder")
    parser.add_argument("openmvs_path", type=Path, help="Path to the OpenMVS binary folder")
    parser.add_argument("img_dir", type=Path, help="Directory containing input images")
    parser.add_argument("out_dir", type=Path, help="Output directory")
    parser.add_argument("scaled_calibration_path", type=Path, help="Path to the (correctly scaled) calibration file")
    parser.add_argument("--mvs_size", type=int, default=2500, help="Maximum resolution for MVS reconstruction")

    args = parser.parse_args()

    reconstruct_3d(
        openmvg_path=args.openmvg_path,
        openmvs_path=args.openmvs_path,
        out_dir=args.out_dir,
        img_dir=args.img_dir,
        scaled_calibration_path=args.scaled_calibration_path,
        mvs_size=args.mvs_size
    )

    # python.exe .\reconstruct_3d.py C:\Users\Admin\Desktop\openmvg_openmvs\openMVG\build\Windows-AMD64-Release\Release C:\Users\Admin\Desktop\openmvg_openmvs\openMVS_sample-0.7a\openMVS_sample-0.7a F:\FIP-data\images\2023\WW034\debayered\2023_06_08_13_11_Lot1\FPWW0340091_FIP2_20230608_122303 F:\FIP-data\images\2023\WW034\debayered\2023_06_08_13_11_Lot1\FPWW0340091_FIP2_20230608_122303\tmp C:\Users\Admin\Desktop\master_thesis\volume_prediction_fip\assets\poses_unscaled\2023_06_08_13_11_Lot1.json