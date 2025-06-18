import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"


import cv2
def resize_and_pad_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    h,w,c = img.shape
    print(h,w)
    if w > h:
        assert w==2048, "Image max size is not 2048"
        assert h==1536, "Image max size is not 1536"
        img = img[:1364,:,:]
    else:
        assert h==2048, "Image max size is not 2048"
        assert w==1536, "Image max size is not 1536"
        img = img[:,:1364,:]
    return img

def resize_exr(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    h,w,c = img.shape
    print(h,w)
    if w > h:
        assert w==2048, "Image max size is not 2048"
        assert h==1536, "Image max size is not 2048"
        img = cv2.resize(img, (2048, 1364))
    else:
        assert h==2048, "Image max size is not 2048"
        assert w==1536, "Image max size is not 2048"
        img = cv2.resize(img, (1364,2048), interpolation=cv2.INTER_AREA)
    return img

if __name__=="__main__":
    import glob
    ext = "*.hdr"
    img_paths = "/home/singh_a_WMGDS.WMG.WARWICK.AC.UK/wmg/itmo/metrices/STAR/fairchild/eval_deep_chain_hdr"
    for img_path in glob.glob(os.path.join(img_paths, ext)):
        img = resize_and_pad_image(img_path)
        name = img_path.split("/")[-1]
        print(img.shape)
        cv2.imwrite(os.path.join("/home/singh_a_WMGDS.WMG.WARWICK.AC.UK/wmg/itmo/metrices/STAR/fairchild/temp", name), img)
    # img_path = "/home/singh_a_WMGDS.WMG.WARWICK.AC.UK/wmg/itmo/metrices/STAR/fairchild/eval_lanet2/507_out.hdr"
    # img = resize_and_pad_image(img_path)
    # print(img.shape)
    # # cv2.imwrite("/home/singh_a_WMGDS.WMG.WARWICK.AC.UK/wmg/itmo/metrices/STAR/507_dh.exr", img)