from train import *
import cv2

if __name__ == "__main__":

    USE_CUDA = bool(torch.cuda.is_available())

    device0 = torch.device("cuda:0" if USE_CUDA else "cpu")
    model = generator(1, 1, device=device0, use_cuda=USE_CUDA)
    #  model.generatorNet.load_state_dict(torch.load('../model_weights/unet-canny-10-20.pth'))

    test_image = cv2.imread('../line_images/P100090.jpg', 0).astype(np.float32)

    #  print(test_image[100:200, 200:400])

    #  generated_image = 1/(1+np.exp(-model.generate(test_image).cpu().numpy()))
    generated_image = model.generate(test_image).cpu().numpy()

    #  print(generated_image.reshape((generated_image.shape[2], -1)).shape)
    print(generated_image.reshape(
        (generated_image.shape[2], -1))[100:200, 200:400])

    cv2.imwrite('./generated_image.jpg',
                generated_image.reshape((generated_image.shape[2], -1))*255.0)
