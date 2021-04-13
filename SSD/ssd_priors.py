# https://medium.com/@smallfishbigsea/understand-ssd-and-implement-your-own-caa3232cd6ad
# 

import collections
import numpy as np
import itertools

SSDBoxSizes = collections.namedtuple('SSDBoxSizes', ['min', 'max'])

Spec = collections.namedtuple('Spec', ['feature_map_size', 'shrinkage', 'box_sizes', 'aspect_ratios'])

# the SSD orignal specs
specs = [
    Spec(38, 8, SSDBoxSizes(30, 60), [2]),
    Spec(19, 16, SSDBoxSizes(60, 111), [2, 3]),
    Spec(10, 32, SSDBoxSizes(111, 162), [2, 3]),
    Spec(5, 64, SSDBoxSizes(162, 213), [2, 3]),
    Spec(3, 100, SSDBoxSizes(213, 264), [2]),
    Spec(1, 300, SSDBoxSizes(264, 315), [2])
]

# the comments will be for Spec(38, 8, SSDBoxSizes(30, 60), [2])
def generate_ssd_priors(specs, image_size=300, clip=True):
    """Generate SSD Prior Boxes.
    
    Args:
        specs: Specs about the shapes of sizes of prior boxes. i.e.
            specs = [
                Spec(38, 8, SSDBoxSizes(30, 60), [2]),
                Spec(19, 16, SSDBoxSizes(60, 111), [2, 3]),
                Spec(10, 32, SSDBoxSizes(111, 162), [2, 3]),
                Spec(5, 64, SSDBoxSizes(162, 213), [2, 3]),
                Spec(3, 100, SSDBoxSizes(213, 264), [2]),
                Spec(1, 300, SSDBoxSizes(264, 315), [2])
            ]
        image_size: image size.
    
    Returns:
        priors: a list of priors: [[center_x, center_y, h, w]]. All the values
            are relative to the image size (300x300).
    """
    boxes = []
    for spec in specs:
        scale = image_size / spec.shrinkage  # 300 / 8 = 37.5
        for j, i in itertools.product(range(spec.feature_map_size), repeat=2):  # create 38 x 38 center points
            x_center = (i + 0.5) / scale
            y_center = (j + 0.5) / scale

            # small sized square box
            size = spec.box_sizes.min  # size is 30
            h = w = size / image_size  # 30 / 300 = 0.1
            boxes.append([
                x_center,    # ex: 0.5
                y_center,    # ex: 0.5
                h,           # ex: 0.1
                w            # ex: 0.1
            ])
            
            # big sized square box
            size = np.sqrt(spec.box_sizes.max * spec.box_sizes.min) # sqrt(60 * 30 ) = 42.43
            h = w = size / image_size   # 42.43 / 300 = 0.1414
            boxes.append([
                x_center,   # ex: 0.5
                y_center,   # ex: 0.5
                h,          # ex: 0.1414
                w           # ex: 0.1414
            ])           
            
            # change h/w ratio of the small sized box
            # based on the SSD implementation, it only applies ratio to the smallest size.
            # it looks wierd.
            size = spec.box_sizes.min # 30
            h = w = size / image_size # 30 / 300 = 0.1
            for ratio in spec.aspect_ratios:
                ratio = np.sqrt(ratio)   # sqrt(2)               
                boxes.append([
                    x_center,  # ex: 0.5
                    y_center,  # ex:0.5
                    h * ratio, # ex: 0.1 * sqrt(2) 
                    w / ratio  # ex: 0.1 / sqrt(2)
                ])
                boxes.append([
                    x_center,  # ex: 0.5
                    y_center,  # ex: 0.5
                    h / ratio, # ex: 0.1 / sqrt(2)
                    w * ratio  # ex:0.1 * sqrt(2)
                ])
            


    boxes = np.array(boxes)
    if clip:
        boxes = np.clip(boxes, 0.0, 1.0)
    return boxes