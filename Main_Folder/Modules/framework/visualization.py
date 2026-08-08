# a list of fns to visualize results and images
import matplotlib.pyplot as plt
import torch
import SimpleITK as sitk
import itk
from airlab.utils import Image as AirlabImage
import numpy as np
from pathlib import Path
from PIL import Image as PImage
from typing import Union, Optional, Sequence
from Main_Folder.Modules.configuration_setting.logger_configuration import get_logger
from Main_Folder.Modules.configuration_setting.yaml_configuration import FrameworkConfig
from Main_Folder.Modules.utils import permute_channel_layout, get_data_time

Image_Type = Union[torch.Tensor, itk.Image, sitk.Image, AirlabImage, Path, np.ndarray]

standard_log = get_logger(__name__)
def _image_to_numpy(image : Image_Type
                    )-> np.ndarray:
    '''
    Asimple function to convert any image type in np.ndarray to plot using matplotlib

    Args:
        image (Image_Type): the image to be convert in a np.ndarray

    Returns:
        np.ndarray: the image as np.ndarray 
    '''
    if isinstance(image, torch.Tensor):
        if image.squeeze().ndim == 3:
            return permute_channel_layout(image=image.squeeze(), target_format = '**C').detach().cpu().numpy()
        return image.squeeze().detach().cpu().numpy()
        
    elif isinstance(image, itk.Image):
        return itk.array_from_image(image)

    elif isinstance(image, sitk.Image):
        return sitk.GetArrayFromImage(image=image)

        
    elif isinstance(image, AirlabImage):
        return image.image.detach().cpu().numpy()
        
    elif isinstance(image, Path):
        return np.asarray(PImage.open(image))
    elif isinstance(image, np.ndarray):
        standard_log.info('the image type is already a np.ndarray, so it\'returned as it is')
        return image
    else:
        standard_log.error(f'unable to convert since: f{type(image)} is yet to be handled')
        raise TypeError (f'Unsupported Type: {type(image)}')

    
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
        img = _image_to_numpy(images[0])
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
        img = _image_to_numpy(image)
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


def show_image_and_reference_points(image: torch.Tensor,
                                    coordinates: list[list[float]],                                    
                                    save_title : str = None,
                                    uniform_color : bool = True,
                                    ):
    ''''''

    
    x_test = [test_point[0] for test_point in coordinates]
    y_test = [test_point[1] for test_point in coordinates]
    result_image_folder = FrameworkConfig().path_dict['RESULTS'] /'ImgRes'
    plt.imshow(image.permute(1,2,0).detach().cpu().numpy())
    plt.axis('off')
    if not uniform_color:
        colors = plt.cm.jet(np.linspace(0, 1, len(coordinates)))
    else: 
        colors = ['black'] * len(coordinates)
    plt.scatter(x_test, y_test, c=colors, s=1)
    if save_title is not None:
        save_path=  Path(result_image_folder / f'{save_title}.png')
        if save_path.exists():
            save_path= Path(f'{save_path.split('.'[0])}_{get_data_time()}.png')
        plt.savefig(save_path)
        
    