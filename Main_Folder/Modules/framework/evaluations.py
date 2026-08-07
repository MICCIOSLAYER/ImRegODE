# use for evaluate the efficiency of certain methods of registration: as NAED and similar
# it contains also the utils for these methods
from Main_Folder.Modules.framework.data_classes import SampleDict
from Main_Folder.Modules.configuration_setting.yaml_configuration import  FrameworkConfig
from typing import Optional, Union
from pathlib import Path




def get_coords(image_name : str | SampleDict , #es A01
               ground_truth_path : Path = FrameworkConfig().path_dict['CONTROL_POINTS_FOLDER'], # in this case: control_points_[Image pair name]_1_2.txt
               )-> tuple[list[list[float]], list[list[float]]]:
    '''
    get the coordinates of control points associate to the image in list of coords format
    the format in the ground truth folder:

        [reference_point_1_x] [reference_point_1_y] [test_point_1_x] [test_point_1_y]

        [reference_point_2_x] [reference_point_2_y] [test_point_2_x] [test_point_2_y]

    it will returns the list of coords for both test(_2) and reference(_1) image in the format test(x,y) ; reference(x,y)
    '''
    if not isinstance(image_name, str):
        name = image_name['sample_name']
        image_name = name
    if not ground_truth_path.exists():
        raise FileNotFoundError(f'directory {ground_truth_path} not found')
    txt_file_path = [n for n in list(ground_truth_path.glob('*')) if n.name.__contains__(image_name)]
    reference_points, test_points = [], []
    with open(txt_file_path[0], 'r') as file:
        lines = file.readlines()
        for line in lines:
            ref_x, ref_y, test_x, test_y = line.split()
            reference_points.append([float(ref_x), float(ref_y)])
            test_points.append([float(test_x), float(test_y)])
    return test_points, reference_points
