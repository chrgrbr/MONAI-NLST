import SimpleITK
import torch
import numpy
from monai.networks.nets import SegResNet
from monai.utils import first
from monai.transforms import (
    Compose,
    LoadImage,
    Resize,
    ScaleIntensityRange,
    ToDevice,
)
import warnings
from pathlib import Path
#pip install -q "monai-weekly[nibabel, tqdm]" itk
# Installing the recommended dependencies 
# https://docs.monai.io/en/stable/installation.html


class MONAI_Registration_NLST():  # SegmentationAlgorithm is not inherited in this class anymore
    def __init__(self):

        self.in_path = Path('/input/images')
        self.out_path = Path('/output/images/displacement-field')
        ##create displacement output folder 
        self.out_path.mkdir(parents=True, exist_ok=True)

        self.model_path = Path('/opt/algorithm/segresnet_kpt_loss_best_tre_1980_0.533.pth')  



        self.target_res = [224, 192, 224]
        self.spatial_size = [-1, -1, -1]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('Using Device:', self.device)


        self.model = SegResNet(
            spatial_dims=3,
            in_channels=2,
            out_channels=3,
            blocks_down=[1, 2, 2, 4],
            blocks_up=[1, 1, 1],
            init_filters=16,
            dropout_prob=0.2,
        )

        self.model.load_state_dict(torch.load(str(self.model_path), map_location=self.device))
        print('Model loaded')
        self.model.eval()
        self.model.to(self.device)
 

        self.transform_image = Compose(
        [
            LoadImage(ensure_channel_first=True, image_only=True, reader='itkreader'),
            ToDevice(self.device),
            ScaleIntensityRange(a_min=-1200, a_max=400, b_min=0.0, b_max=1.0, clip=True),
            Resize(
                mode=("trilinear"),
                align_corners=(True),
                spatial_size=self.target_res,
            )
        ])
        return


    def load_inputs(self):

        ## Grand Challenge Algorithms expect only one file in each input folder, i.e.:
        fpath_fixed_image = list((self.in_path / 'fixed').glob('*.mha'))[0]
        fpath_moving_image = list((self.in_path / 'moving').glob('*.mha'))[0]
        fpath_fixed_mask = list((self.in_path / 'fixed-mask').glob('*.mha'))[0]
        fpath_moving_mask = list((self.in_path / 'moving-mask').glob('*.mha'))[0]
        print('Fixed Image:', fpath_fixed_image)
        print('Moving Image', fpath_moving_image)

        #We do not need masks for this algorithm.


        fixed_image = self.transform_image(str(fpath_fixed_image))
        moving_image = self.transform_image(str(fpath_moving_image))
        return fixed_image, moving_image

    def write_outputs(self, displacement_field):
        displacement_field = displacement_field.squeeze(0).permute(1, 2, 3, 0).cpu().numpy()
        out = SimpleITK.GetImageFromArray(displacement_field)
        ##You can give the output-mha file any name you want, but it must be in the /output/displacement-field folder
        SimpleITK.WriteImage(out, str(self.out_path / 'thisIsAnArbitraryFilename.mha'))


        return
    
    def predict(self, inputs):
        # Predict the displacement field
        fixed_image, moving_image = inputs
        with torch.inference_mode():
            displacement_field = self.model(torch.cat((moving_image, fixed_image), dim=0).unsqueeze(0)).float()
        return displacement_field



    def process(self):

        inputs = self.load_inputs()
        outputs = self.predict(inputs)
        self.write_outputs(outputs)

        return

if __name__ == "__main__":
    MONAI_Registration_NLST().process()
