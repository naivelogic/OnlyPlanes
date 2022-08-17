import albumentations as A

#FutureWarning: Cutout has been deprecated. Please use CoarseDropout
Ax03_random_custouts = [
            A.Cutout(num_holes=15,max_h_size=12,max_w_size=12,fill_value=0,p=0.5),#black
            A.Cutout(num_holes=15,max_h_size=12,max_w_size=12,fill_value=255,p=0.3),#white
            A.Cutout(num_holes=15,max_h_size=12,max_w_size=12,fill_value=120,p=0.3),#gray
            ] ## other from Ax03

def Ax08(train_method): 
    aug_list = [
        A.Resize(1024,1024,p=1), # Onlyy for DSmw
        A.HorizontalFlip(p=0.3),
        A.VerticalFlip(p=0.7),

        A.OneOf([
            A.RandomSizedCrop(min_max_height=(800,1024),height=512,width=512,p=0.8),
            A.RandomSizedCrop(min_max_height=(256,800),height=512,width=512,p=0.4),
            A.RandomSizedBBoxSafeCrop(512,512,erosion_rate=-2,p=0.5),
        ],p=1),

        A.OneOf([
            #A.RandomScale(scale_limit=(-0.8, -0.8), p=0.4),
            A.RandomScale(scale_limit=(-0.3, 0.1), p=0.4),
            A.MultiplicativeNoise((0.9,1.5),False,False,p=0.5), # great noise addition
        ],p=0.6),

        # collor variation from Ax07
        A.OneOf([
            A.ChannelShuffle(p=0.3), 
            A.RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(0,0), p=0.3), 
            A.RandomBrightnessContrast(brightness_limit=(0,0), contrast_limit=(-0.2,0.2), p=0.3), 
            A.ToGray(p=0.1),
            A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.4),
            A.HueSaturationValue(hue_shift_limit=10,sat_shift_limit=10,val_shift_limit=0,p=0.4)
        ],p=0.8),

        # image compression Ax07
        A.OneOf([
            A.ImageCompression(20,80,p=0.6),
            #A.JpegCompression(50,50,p=0.6),
            A.Downscale(0.4,0.7,p=0.5),
            A.Posterize(num_bits=[5,8], p=0.25), #Reduce the number of bits for each color channel. | 2 is really extreme
        ],p=0.85),
        
        ## other from Ax03
        A.OneOf(Ax03_random_custouts,p=0.3),
        
    ]

    if train_method == 'bbox':
        return A.Compose(aug_list, bbox_params=A.BboxParams(format="coco", min_visibility=0.15))
    elif train_method == 'segm':
        return A.Compose(aug_list, bbox_params=A.BboxParams(format="coco", label_fields=['category_id','bbox_ids'], min_visibility=0.15))