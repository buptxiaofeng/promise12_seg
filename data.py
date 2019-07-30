import SimpleITK as sitk
from scipy import ndimage
import glob
import os
import numpy

class Data:
    def __init__(self):
        pass

    def load_train_data(self, path, new_spacing = [1, 1, 1.5], volume_size = [128, 128, 64]):
        path = os.path.join(path, "*.mhd")
        mhd_list = glob.glob(path)
        image_list = [x for x in mhd_list if "segment" not in x]
        mask_list = [x for x in mhd_list if "segment" in x]
        image_list = sorted(image_list)
        mask_list = sorted(mask_list)

        image_result_list = []
        mask_result_list = []

        for image, mask in zip(image_list, mask_list):
            mhd_image = sitk.ReadImage(image)
            mhd_image = self.rescale(mhd_image, new_spacing = new_spacing)
            mhd_image = self.crop(mhd_image)
            mhd_image = self.normalize(mhd_image)
            mhd_image = sitk.GetArrayFromImage(mhd_image)
            mhd_mask = sitk.ReadImage(mask)
            mhd_mask = sitk.Cast(mhd_mask > 0.5, sitk.sitkFloat32)
            mhd_mask = self.rescale(mhd_mask, new_spacing)
            mhd_mask = self.crop(mhd_mask)
            mhd_mask = sitk.GetArrayFromImage(mhd_mask)
            mhd_image = numpy.reshape(mhd_image, (1, mhd_image.shape[0], mhd_image.shape[1], mhd_image.shape[2]))
            mhd_mask = numpy.reshape(mhd_mask, (1, mhd_mask.shape[0], mhd_mask.shape[1], mhd_mask.shape[2]))
            image_result_list.append(mhd_image)
            mask_result_list.append(mhd_mask)

        image_data = numpy.asarray(image_result_list)
        mask_data = numpy.asarray(mask_result_list)

        return image_data, mask_data

    def rescale(self, mhd_image, new_spacing = [1, 1, 1.5], volume_size = [128, 128, 64]):
        factor = numpy.asarray(mhd_image.GetSpacing()) / new_spacing
        new_size = numpy.asarray(mhd_image.GetSize() * factor, dtype=numpy.int16)
        new_size = numpy.max([new_size, volume_size], axis = 0)
        resample_filter = sitk.ResampleImageFilter()
        resample_filter.SetReferenceImage(mhd_image)
        resample_filter.SetOutputSpacing(new_spacing)
        resample_filter.SetSize(new_size.tolist())
        resample_filter.SetInterpolator(sitk.sitkLinear)
        mhd_image = resample_filter.Execute(mhd_image)

        return mhd_image

    def crop(self, mhd_image, volume_size = [128, 128, 64]):
        new_size = numpy.max([mhd_image.GetSize(), volume_size], axis=0)
        image_centroid = numpy.asarray(new_size, dtype = float) / 2.0
        image_start_pixel = (image_centroid - numpy.array(volume_size) / 2.0).astype(dtype="int")
        region_extractor = sitk.RegionOfInterestImageFilter()
        region_extractor.SetSize(volume_size)
        region_extractor.SetIndex(image_start_pixel.tolist())
        mhd_image = region_extractor.Execute(mhd_image)

        return mhd_image

    def normalize(self, mhd_image):
        normal_filter = sitk.RescaleIntensityImageFilter()
        normal_filter.SetOutputMaximum(1)
        normal_filter.SetOutputMinimum(0)
        m = 0.
        mhd_image = sitk.Cast(mhd_image, sitk.sitkFloat32)
        mhd_image = normal_filter.Execute(mhd_image)

        return mhd_image

    #return image list and path list 
    def load_test_data(self, path, new_spacing = [1, 1, 1.5], volume_size = [128, 128, 64]):
        path = os.path.join(path, "*.mhd")
        mhd_list = glob.glob(path)
        image_result_list = []
        image_path_list = []
        for image in mhd_list:
            mhd_image = sitk.ReadImage(image)
            mhd_image = self.rescale(mhd_image, new_spacing)
            mhd_image = self.crop(mhd_image)
            mhd_image = self.normalize(mhd_image)
            mhd_image = sitk.GetArrayFromImage(mhd_image)
            mhd_image = numpy.reshape(mhd_image, (1, mhd_image.shape[0], mhd_image.shape[1], mhd_image.shape[2]))
            image_path_list.append(image)
            image_result_list.append(mhd_image)

        image_result_list = numpy.asarray(image_result_list)
        image_path_list = numpy.asarray(image_path_list)

        return image_result_list, image_path_list

if __name__ == "__main__":
    train_path = "promise12/train"
    test_path = "promise12/test"
    data = Data()
    #data, label = data.load_train_data(train_path)
    #print(data.shape)
    #print(label.shape)
    data.load_test_data(test_path)
