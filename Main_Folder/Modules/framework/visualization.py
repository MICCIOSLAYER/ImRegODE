# a list of fns to visualize results and images
import matplotlib.pyplot as plt
import torch
import SimpleITK as sitk
import sys
import itk
from airlab.utils import Image as AirlabImage
from airlab.transformation.utils import warp_image

import numpy as np
from pathlib import Path
from datetime import datetime
from PIL import Image as PImage
from typing import Union, Optional, Sequence, Any
from Main_Folder.Modules.configuration_setting.logger_configuration import get_logger
from Main_Folder.Modules.configuration_setting.yaml_configuration import FrameworkConfig
from Main_Folder.Modules.framework.preprocessing import rescaling_image
from Main_Folder.Modules.utils import permute_channel_layout, get_data_time, get_root_path, concatenate_paths, image_to_numpy

Image_Type = Union[torch.Tensor, itk.Image, sitk.Image, AirlabImage, Path, np.ndarray]

standard_log = get_logger(__name__)


    
def show_images(image_list: Sequence[Image_Type] | Image_Type,
                image_title_list : Sequence[str] | str | None = None,
                images_for_row : int = 3,
                cmap: str = 'gray',
                fig_size : tuple[int, int] = (12, 8),
                ): # FIXME add a flag to save single images
    '''
    A simple function to plot images as subplot if the input is a a list, or simply show a single image if single

    Args:
        image_list (list[Image_Type] | Image_Type): The  input/-s to be show
        image_title_list (Optional[list[str]] | str, optional): the title for each image. Defaults to None.
        images_for_row (int): the max number of images for each row in a subplot. Default to 3
        cmap (str): The map whose images in grayscale will follow. Default to 'gray'
        fig_size (tuple[int,int]): the size of images. Default to (12,8)


    '''
    if isinstance(image_list, Sequence) and not isinstance(image_list, (Path, str)):
        images = list(image_list)
    else:
        images = [image_list]
    n_images= len(images)

    if image_title_list is None:
        titles = [f'img_{i}' for i in range(n_images)]
    elif isinstance(image_title_list, str):
        titles= [image_title_list]
    else:
        titles = list(image_title_list)

    if len(titles) != n_images:
        raise ValueError('the number of titles must match the number of images')

    if n_images == 1:
        img = image_to_numpy(images[0])
        plt.figure(figsize=fig_size)
        if img.ndim == 2:
            plt.imshow(img, cmap=cmap)
        else:
            plt.imshow(img)
        plt.axis('off')
        plt.title(titles[0])
        plt.tight_layout()
        plt.show()
        return 

    n_cols = min (images_for_row, n_images)
    n_row = (n_images +n_cols -1)// n_cols
    fig, axes = plt.subplots(n_row, n_cols, figsize= fig_size)
    axes = np.atleast_1d(axes).ravel()
    for ax, image, title in zip(axes, images, titles):
        img = image_to_numpy(image)
        if img.ndim == 2:
            ax.imshow(img, cmap=cmap)
        else:
            ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")

    # Hide unused axes
    for ax in axes[n_images:]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()

    return 


def show_image_and_reference_points(image: Image_Type,
                                    coordinates: list[list[float]],                                    
                                    save_title : str = None,
                                    uniform_color : bool = True,
                                    )-> Optional[Path]:
    '''
    Show the image with the reference points over it, and save it if a title is provided

    Args:
        image (Image_Type): the image to show
        coordinates (list[list[float]]): the coordinates of ground truth points to show over the image
        save_title (str, optional): the title to save the image. Defaults to None.
        uniform_color (bool, optional): whether to use a uniform color for all points. Defaults to True.
        
    '''

    
    x_test = [test_point[0] for test_point in coordinates]
    y_test = [test_point[1] for test_point in coordinates]
    
    plt.imshow(image_to_numpy(image=image))
    plt.axis('off')
    if not uniform_color:
        colors = plt.cm.jet(np.linspace(0, 1, len(coordinates)))
    else: 
        colors = ['black'] * len(coordinates)
    plt.scatter(x_test, y_test, c=colors, s=1)
    if save_title is not None:
        result_image_folder = get_root_path() / 'Results' / 'ImgRes'
        save_path=  Path(result_image_folder / f'{save_title}.png')
        if save_path.exists():
            save_path= Path(f'{save_path.split('.'[0])}_{get_data_time()}.png')
        plt.savefig(save_path)
        return save_path
    plt.show()
    return


def elastix_show_difference_image(reference_image: itk.Image,
                                test_image: itk.Image,
                                registered_image: itk.Image,
                                saving_path: Path = None,                                
                               )-> Optional[Path]:
    '''
        Show the difference images between reference and test, and reference and registered images.

    Args:
        reference_image (itk.Image): image of reference
        test_image (itk.Image): image of testing
        registered_image (itk.Image): image warped through registration
        saving_path (Path, optional): saving path. Defaults to None.

    Returns:
        Path :  saving path. Defaults to None.
    '''
    reference_normalized = rescaling_image(reference_image)
    test_normalized = rescaling_image(test_image)
    registered_normalized = rescaling_image(registered_image)
    original_difference = np.array(itk.array_view_from_image(reference_normalized) - np.array(itk.array_view_from_image(test_normalized)))
    registered_difference = np.array(itk.array_view_from_image(reference_normalized) - np.array(itk.array_view_from_image(registered_normalized)))
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    plt.figsize=[100,100]
    axes[0].imshow(original_difference, cmap='gray')
    axes[0].set_title('Prior Registration', fontsize=10, loc='left')
    axes[0].axis('off')
    axes[0].set_title(f'RMSE: {np.sqrt((original_difference**2).mean()):.4f}', fontsize=10, loc='right')
    axes[1].imshow(registered_difference, cmap='gray')
    axes[1].set_title('After Registration', fontsize=10, loc='left')
    axes[1].axis('off')
    axes[1].set_title(f'RMSE: {np.sqrt((registered_difference**2).mean()):.4f}', fontsize=10, loc='right')
    fig.get_label()
    if saving_path.exists():
        imagedir_save_path = saving_path
    elif not saving_path.exists() and not saving_path is None:
        imagedir_save_path = FrameworkConfig().path_dict['IMG_RESULTS']
        standard_log.warning(f'since the chosen {saving_path} do not corresponds to any of existent path the default one in yaml configuration is used')
    else:
        return None
    saving_name = Path(imagedir_save_path) / 'Difference_Elastix.png'
    if saving_name.exists():
        saving_name = Path(imagedir_save_path) / f'Difference_Elastix{get_data_time()}.png'
    plt.savefig(saving_name)
    plt.show()

    return imagedir_save_path


def airlab_show_image_differencies(reference_image: AirlabImage,
                                   test_image: AirlabImage,
                                   airlab_transformation: Union[torch.nn.Module, Any],
                                   config_dict : dict | FrameworkConfig,
                                   save_images: Union[Path, bool] = False,
                                   
                                   )->Optional[Path]:
    '''
    Shoe & Save images confrontation to get a visual impact on results

    Args:
        reference_image (AirlabImage): ground truth of registration
        test_image (AirlabImage): the deformed image, to be registred or to be warped
        airlab_transformation (Union[nn.Module, Any]): basically the warping transformation obatined by the registration
        save_images (Union[bool, Path]): Define if save it or not in a specified path otherwise it will be placed in ImRes folder inside repos. Defaults to False.
        config_dict: dict | FrameworkConfig: the space in which there are the information on metric sigma, n_histo_bin, n_iteration, lr

    Returns:
        Optional[Path]: the path of the saved image if save_images is True, otherwise None
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    IN_COLAB = 'google.colab' in sys.module
    if isinstance(config_dict, dict):
        airlab_dict=config_dict
        imagedir_save_path = config_dict.get('save_path', save_images)
    else:
        airlab_dict= config_dict.registrations['airlab']
        imagedir_save_path = config_dict.path_dict['IMG_RESULTS']

    metric_sigma = airlab_dict['metric_sigma']
    metric_num_bins = airlab_dict['histo_bins']
    num_iterations= airlab_dict['n_iterations']
    learning_rate= airlab_dict['lr']
    PROJECT_PATH = get_root_path()

    


    prior_reg = np.abs(reference_image.numpy() - test_image.numpy())
    if hasattr(airlab_transformation, 'get_displacement'):
        displacement_field = airlab_transformation.get_displacement()
        after_reg = np.abs(reference_image.numpy() - warp_image(test_image, displacement_field).numpy())
    #========== RESULTS================ show_save_image
    
    save_time=datetime.now().strftime("%Y%m%d_%H%M%S")
    save_name = f'AMI_sigma{metric_sigma}_niter{num_iterations}_bins{metric_num_bins}_lr{learning_rate}__{save_time}.png'
    
    
    
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].set_title('Original Difference')
    ax[0].imshow(prior_reg, cmap='gray')
    ax[0].axis('off')
    ax[1].set_title('Warped Difference')
    ax[1].imshow(after_reg, cmap='gray')
    ax[1].axis('off')

   

    if isinstance(save_images, Path) and save_images.exists():
        saving_path = save_images / save_name
    elif not save_images:
        return saving_path
    else:
        saving_path = Path(imagedir_save_path) / save_name



    if IN_COLAB:
        saving_path = Path(f'/content/drive/MyDrive/Results/{save_name}')
        if saving_path.exists():
            saving_path = saving_path.parent / f'{save_time}.png'
        fig.savefig(saving_path,
                    dpi=150,
                    bbox_inches='tight',
                    pad_inches=0.1)
    else:
        if saving_path.exists():
            saving_path = saving_path.parent / f'{save_time}.png'
        fig.savefig(saving_path)
    plt.show()
    plt.close(fig)
        
    
    return saving_path