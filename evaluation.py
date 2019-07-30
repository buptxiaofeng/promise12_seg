import torch
import numpy
from dataset import PromiseDataset
from data import Data
from model import VNet
import torch.backends.cudnn as cudnn
import SimpleITK as sitk
import os

def write_result(result, path, new_spacing = [1, 1, 1.5], volume_size = [128, 128, 64]):
    original_image = sitk.ReadImage(path)
    to_write = sitk.Image(original_image.GetSize()[0], original_image.GetSize()[1], original_image.GetSize()[2], sitk.sitkFloat32)
    factor = numpy.asarray(original_image.GetSpacing()) / [new_spacing[0], new_spacing[1], new_spacing[2]]
    factor_size = numpy.asarray(original_image.GetSize() * factor, dtype=float)
    new_size = numpy.max([factor_size, volume_size], axis = 0)
    new_size = new_size.astype(dtype=int)
    
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(original_image)
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size.tolist())
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    
    to_write = resampler.Execute(to_write)
    image_centroid = numpy.asarray(new_size, dtype = float) / 2.0
    image_start_pixel = (image_centroid - numpy.asarray(volume_size) / 2.0).astype(dtype = int)

    for dst_x, src_x in zip(range(0, result.shape[0]), range(image_start_pixel[0], int(image_start_pixel[0] + volume_size[0]))):
        for dst_y, src_y in zip(range(0, result.shape[1]), range(image_start_pixel[1], int(image_start_pixel[1] + volume_size[1]))):
            for dst_z, src_z in zip(range(0, result.shape[2]), range(image_start_pixel[2], int(image_start_pixel[2] + volume_size[2]))):
                to_write.SetPixel(int(src_x), int(src_y), int(src_z), float(result[dst_x, dst_y, dst_z]))
    
    resampler.SetOutputSpacing([original_image.GetSpacing()[0], original_image.GetSpacing()[1], original_image.GetSpacing()[2]])
    resampler.SetSize(original_image.GetSize())
    to_write = resampler.Execute(to_write)
    threshold_filter = sitk.BinaryThresholdImageFilter()
    threshold_filter.SetInsideValue(1)
    threshold_filter.SetOutsideValue(0)
    threshold_filter.SetLowerThreshold(0.5)
    to_write = threshold_filter.Execute(to_write)
    
    connected_component = sitk.ConnectedComponentImageFilter()
    to_write_connected_component = connected_component.Execute(sitk.Cast(to_write, sitk.sitkUInt8))
    connected_component_array = numpy.transpose(sitk.GetArrayFromImage(to_write_connected_component).astype(dtype = float), [2, 1, 0])
    lab = numpy.zeros(int(numpy.max(connected_component_array) + 1), dtype = float)
    for i in range(1, int(numpy.max(connected_component_array) + 1)):
        lab[i] = numpy.sum(connected_component_array == i)
    active_lab = numpy.argmax(lab)
    to_write = (to_write_connected_component == active_lab)
    to_write = sitk.Cast(to_write, sitk.sitkUInt8)
    writer = sitk.ImageFileWriter()
    patient_name = path.split("/")[-1]
    writer.SetFileName(os.path.join("result", patient_name))
    writer.Execute(to_write)

def inference(model):
    model.eval()
    promise_dataset = PromiseDataset(is_train = False)
    test_loader = torch.utils.data.DataLoader(dataset = promise_dataset, batch_size = 1)
    with torch.no_grad():
        for batch_index, data in enumerate(test_loader):
            data, path = data
            pred = model(data)
            _, _, z, y, x = data.shape
            result = model(data)
            #tuple to string
            path = "".join(path)
            write_result(result, path)

if __name__ == "__main__":
    net = VNet()
    net = torch.nn.DataParallel(net, device_ids=[0, 1, 2, 3])
    net.cuda()
    cudnn.benchmark = True
    net.load_state_dict(torch.load("weights/promise12_weight.pth"))
    inference(net)
