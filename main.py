import onnxruntime as ort
import numpy as np
import matplotlib.pyplot as plt
import wget, time, os, urllib
import torchvision
import torchvision.transforms as transforms

def download_model(model_name):
    if not os.path.exists(model_name):
        base_url = 'https://github.com/onnx/models/blob/main/validated/vision/classification/mnist/model/'
        url = urllib.parse.urljoin(base_url, model_name)
        wget.download(url)

def onnx_predict(onnx_session, input_name, output_name,  
    test_images, test_labels, image_index, show_results): 
  
    test_image = np.expand_dims(test_images[image_index], [0,1]) 
  
    onnx_pred = onnx_session.run([output_name], {input_name: test_image.astype('float32')}) 
  
    predicted_label = np.argmax(np.array(onnx_pred)) 
    actual_label = test_labels[image_index] 
  
    if show_results: 
        plt.figure() 
        plt.xticks([]) 
        plt.yticks([])  
        plt.imshow(test_images[image_index], cmap=plt.cm.binary) 
         
        plt.title('Actual: %s, predicted: %s'  
            % (actual_label, predicted_label), fontsize=22)                
        plt.show() 
     
    return predicted_label, actual_label 

def measure_performance(onnx_session, input_name, output_name,  
    test_images, test_labels, execution_count): 
 
    start = time.time()     
  
    image_indices = np.random.randint(0, test_images.shape[0] - 1, execution_count) 
     
    for i in range(1, execution_count): 
        onnx_predict(onnx_session, input_name, output_name,  
            test_images, test_labels, image_indices[i], False) 
     
    computation_time = time.time() - start 
     
    print('Computation time: %.3f ms' % (computation_time*1000)) 

if __name__ == "__main__":
    # Download and prepare the model
    model_name = 'mnist-12.onnx'
    download_model(model_name)

    # Set up ONNX inference session
    onnx_session = ort.InferenceSession(model_name)

    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name

    # Load the MNIST dataset using torchvision
    transform = transforms.Compose([transforms.ToTensor()])
    mnist_dataset = torchvision.datasets.MNIST(root='./data', train=False,
                                               download=True, transform=transform)

    test_images = mnist_dataset.data.numpy()
    test_labels = mnist_dataset.targets.numpy()

    # Normalize images
    test_images = test_images / 255.0

    # Perform a single prediction and display the result
    image_index = np.random.randint(0, test_images.shape[0] - 1)
    onnx_predict(onnx_session, input_name, output_name,  
                 test_images, test_labels, image_index, True)

    # Measure inference performance
    measure_performance(onnx_session, input_name, output_name,  
                        test_images, test_labels, execution_count=1000)