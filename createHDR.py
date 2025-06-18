import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
import numpy as np
import utils.ditmo_utils as du
from VariableParams import *
from glob import glob
from merge_hdr_francesco import *

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

save_exr = "/home/singh_a_WMGDS.WMG.WARWICK.AC.UK/wmg/itmo/AutomateDITMO/outputs/hal/controlNet/exr_fr/"
save_pfm = "/home/singh_a_WMGDS.WMG.WARWICK.AC.UK/wmg/itmo/AutomateDITMO/outputs/hal/controlNet/pfm/"
prompt_dicts = ['pd5']
base_exp_t = [1]
directory = "/home/singh_a_WMGDS.WMG.WARWICK.AC.UK/wmg/itmo/AutomateDITMO/paper_img/"

filenames = [f.split('.')[0] for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

for dict in prompt_dicts:
    addr = f"/home/singh_a_WMGDS.WMG.WARWICK.AC.UK/wmg/itmo/AutomateDITMO/outputs/hal/controlNet/ldr/{dict}/"
    for b in base_exp_t:
        for i, rd in enumerate(morphology_radius_pairs):
            for fn in filenames:
                path = f"{addr}{legend_rad[i]}/{fn}.png/"
                img_list_ldr = []
                img_path_list_ldr = []
                exp_list = []
                print(fn)
                try:
                    for ldrs in glob(path+'*.png'):
                        img_path_list_ldr.append(ldrs)
                        fname = ldrs.split('/')[-1].split('_brack')[0]
                        brack = cv2.imread(ldrs)
                        brack = cv2.cvtColor(brack, cv2.COLOR_BGR2RGB)
                        img_list_ldr.append(brack)
                        brack_num = ldrs.split('/')[-1].split('.')[0].split('_')[-1]
                        exp_t = b / (2 ** int(brack_num))
                        exp_list.append(exp_t)
                    print(exp_list)
 

                    hdr = buildHDRwithPILWithouEXIF(img_path_list_ldr, exp_list)

                    hdr = np.float32(hdr)
                    cv2.imwrite(f"{save_exr}{fname}_debevec_francesco.exr", cv2.cvtColor(hdr, cv2.COLOR_BGR2RGB))
                    du.write_pfm(f"{save_pfm}{fname}_debevec_francesco.pfm", hdr)
                except:
                    print(f"images skipped in{path} ")
                    pass
